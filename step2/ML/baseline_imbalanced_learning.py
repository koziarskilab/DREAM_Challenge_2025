from helper import Dataset, ProcessData
from network import MLP, BBNModel
from mixup import mixup_data, mixup_criterion
from remix import remix_data
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
import datetime
import pickle
import numpy as np
from tqdm import tqdm
from itertools import combinations

# Import the loss functions from the molecular_imbalanced_benchmark
from loss import (
    ClassBalanceFocal,
    BalancedSoftmaxCE,
    ClassBalanceCE,
    CostSensitiveCE,
    InfluenceBalancedLoss,
    CDT,  
)


def concatenate_fingerprints(df, fps_types):
    """
    Concatenate multiple fingerprint types for a given dataframe.
    
    Args:
        df: DataFrame containing the data
        fps_types: List of fingerprint types to concatenate
    
    Returns:
        Concatenated fingerprint features
    """
    concatenated_data = []
    
    for fps_type in fps_types:
        data = ProcessData(df, fps_type).get_data()
        concatenated_data.append(data)
    
    # Concatenate along feature axis (axis=1)
    return np.concatenate(concatenated_data, axis=1)


def create_weighted_sampler(labels, sampling_type="balanced"):
    """Create weighted sampler for different sampling strategies"""
    class_counts = np.bincount(labels)
    num_samples = len(labels)
    
    if sampling_type == "balanced":
        # Inverse frequency weighting
        class_weights = 1.0 / class_counts
        weights = class_weights[labels]
    elif sampling_type == "progressive":
        # Progressive resampling (used in Decoupling)
        # Start with inverse frequency, gradually move to uniform
        class_weights = 1.0 / class_counts
        weights = class_weights[labels]
    else:
        # Uniform sampling
        weights = np.ones(num_samples)
    
    return WeightedRandomSampler(weights, num_samples, replacement=True)


def get_loss_function(imbalanced, num_class_list, device):
    """Initialize the appropriate loss function"""
    num_classes = len(num_class_list)
    
    # Create parameter dictionary required by the loss functions
    para_dict = {
        "num_classes": num_classes,
        "num_class_list": num_class_list,
        "device": device,
        "cfg": {
            "loss": {
                "ClassBalanceFocal": {
                    "BETA": 0.999,
                    "GAMMA": 0.5,
                },
                "ClassBalanceCE": {
                    "BETA": 0.999,
                },
                "CostSensitiveCE": {
                    "GAMMA": 1.0,
                },
                "InfluenceBalancedLoss": {
                    "ALPHA": 1.0,  # Reduced from 1000.0
                },
                "CDT": {
                    "GAMMA": 0.1,  # Adjusted for binary classification
                },
            },
            "train": {
                "two_stage": {
                    "drw": False,
                    "start_epoch": 160,
                }
            }
        }
    }
    
    if imbalanced == "CB_F":
        return ClassBalanceFocal(para_dict)
    elif imbalanced == "BS":
        return BalancedSoftmaxCE(para_dict)
    elif imbalanced == "CB_CE":
        return ClassBalanceCE(para_dict)
    elif imbalanced == "CS":
        return CostSensitiveCE(para_dict)
    elif imbalanced == "IB":
        return InfluenceBalancedLoss(para_dict)
    elif imbalanced == "CDT":
        return CDT(para_dict)
    elif imbalanced in ["CE", "MIXUP", "REMIX", "DECOUPLING", "BBN"]:
        return nn.CrossEntropyLoss(reduction='none' if imbalanced == "REMIX" else 'mean')
    else:
        raise ValueError(f"Unknown loss type: {imbalanced}")


def update_results_csv(parent_dir, model_type, fps_type, prauc, roc_auc, metric,
                       hits_50=None, clusters_50=None, cluster_prauc_50=None,
                       hits_200=None, clusters_200=None, cluster_prauc_200=None,
                       hits_500=None, clusters_500=None, cluster_prauc_500=None):
    """Update the results CSV file with enhanced metrics"""
    # Use combination-specific CSV file name
    if "," in fps_type:
        results_file = os.path.join(parent_dir, "combination_model_results.csv")
    else:
        results_file = os.path.join(parent_dir, "model_results.csv")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = {
        "Timestamp": timestamp,
        "ModelType": model_type,
        "FingerprintType": fps_type,
        "OptimizationMetric": metric,
        "PRAUC": prauc,
        "ROC-AUC": roc_auc,
    }
    
    # Add cluster metrics if provided
    if hits_50 is not None:
        new_data["Hits_Top50"] = hits_50
    if clusters_50 is not None:
        new_data["Clusters_Top50"] = clusters_50
    if cluster_prauc_50 is not None:
        new_data["ClusterPRAUC_Top50"] = cluster_prauc_50
    if hits_200 is not None:
        new_data["Hits_Top200"] = hits_200
    if clusters_200 is not None:
        new_data["Clusters_Top200"] = clusters_200
    if cluster_prauc_200 is not None:
        new_data["ClusterPRAUC_Top200"] = cluster_prauc_200
    if hits_500 is not None:
        new_data["Hits_Top500"] = hits_500
    if clusters_500 is not None:
        new_data["Clusters_Top500"] = clusters_500
    if cluster_prauc_500 is not None:
        new_data["ClusterPRAUC_Top500"] = cluster_prauc_500

    if os.path.exists(results_file):
        df_results = pd.read_csv(results_file)
        df_results = pd.concat([df_results, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df_results = pd.DataFrame([new_data])

    df_results.to_csv(results_file, index=False)
    print(f"Results updated in {results_file}")


def calculate_cluster_metrics(df_val, probabilities, top_n):
    """Calculate cluster-based metrics for top N compounds"""
    all_clusters = df_val[df_val["LABEL"] == 1].drop_duplicates("CLUSTER_LABEL").shape[0]
    
    sorted_indices = probabilities.argsort()[::-1]
    selection = df_val.iloc[sorted_indices[:top_n]].copy()
    selection["Score"] = probabilities[sorted_indices[:top_n]]
    
    hits = selection[selection["LABEL"] == 1]
    n_hits = hits.shape[0]
    clusters = hits.drop_duplicates("CLUSTER_LABEL").shape[0]
    
    # Calculate cluster PRAUC
    cluster_prauc = None
    if clusters > 1:
        cluster_recall = []
        cluster_precision = []
        
        for th in sorted(hits["Score"].unique(), reverse=True):
            found = hits[hits["Score"] >= th].drop_duplicates("CLUSTER_LABEL").shape[0]
            cluster_recall.append(found/all_clusters)
            selected = selection[selection["Score"] >= th].shape[0]
            cluster_precision.append(found/selected if selected > 0 else 0)
        
        if len(cluster_recall) >= 2:
            cluster_prauc = auc(cluster_recall, cluster_precision)
        else:
            cluster_prauc = cluster_precision[0] if cluster_precision else 0
    elif clusters == 1:
        th = hits["Score"].min()
        selected = selection[selection["Score"] >= th].shape[0]
        cluster_prauc = 1/selected if selected > 0 else 0
    else:
        cluster_prauc = 0
    
    return n_hits, clusters, cluster_prauc


def get_adaptive_hidden_dims(input_dim, base_factor=0.5, min_hidden=128, max_hidden=1024):
    """
    Calculate adaptive hidden dimensions based on input dimension
    
    Fingerprint dimensions:
    - MACCS: 167 bits
    - RDK: 2048 bits  
    - AVALON: 2048 bits
    - ATOMPAIR: 2048 bits
    
    Args:
        input_dim: Input feature dimension
        base_factor: Factor to scale first hidden layer (0.5 means half of input_dim)
        min_hidden: Minimum hidden dimension
        max_hidden: Maximum hidden dimension per layer
    
    Returns:
        List of hidden dimensions
    """
    # First layer: proportional to input but within bounds
    first_hidden = max(min_hidden, min(int(input_dim * base_factor), max_hidden))
    
    # Second layer: 3/4 of first layer
    second_hidden = max(min_hidden, int(first_hidden * 0.75))
    
    # Third layer: 1/2 of first layer  
    third_hidden = max(min_hidden, int(first_hidden * 0.5))
    
    return [first_hidden, second_hidden, third_hidden]


def train_model(model, train_loader, val_loader, loss_fn, device, df_val, 
                method_type="CE", num_class_list=None, epochs=200, lr=0.001):
    """
    Standard training loop for most imbalanced learning methods
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    best_val_prauc = 0.0
    best_model_state = None
    patience = 30
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        # Update loss function if it has update method (for methods like Decoupling)
        if hasattr(loss_fn, 'update'):
            loss_fn.update(epoch + 1)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            if method_type == "MIXUP":
                # Mixup data augmentation
                data, targets_a, targets_b, lam = mixup_data(data, target, alpha=1.0)
                outputs = model(data)
                loss = mixup_criterion(loss_fn, outputs, targets_a, targets_b, lam)
            elif method_type == "REMIX":
                # Remix augmentation
                data, targets_a, targets_b, lam = remix_data(data, target, alpha=1.0, kappa=3.0)
                outputs = model(data)
                # For REMIX, we need to handle the loss differently
                loss_a = loss_fn(outputs, targets_a)
                loss_b = loss_fn(outputs, targets_b)
                loss = lam * loss_a.mean() + (1 - lam) * loss_b.mean()
            elif method_type == "IB":
                # Influence-balanced loss needs features
                features = model.get_features(data) if hasattr(model, 'get_features') else data
                outputs = model(data)
                loss = loss_fn(outputs, target, feature=features)
            else:
                # Standard forward pass
                outputs = model(data)
                if hasattr(loss_fn, 'forward'):
                    loss = loss_fn(outputs, target)
                else:
                    loss = loss_fn(outputs, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_train_loss = train_loss / num_batches
        
        # Validation phase
        model.eval()
        val_probabilities = []
        val_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                outputs = model(data)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                val_probabilities.extend(probs.cpu().numpy())
                val_labels.extend(target.numpy())
        
        val_probabilities = np.array(val_probabilities)
        val_labels = np.array(val_labels)
        
        # Calculate validation metrics
        val_prauc = average_precision_score(val_labels, val_probabilities)
        val_roc_auc = roc_auc_score(val_labels, val_probabilities)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val PRAUC: {val_prauc:.4f}, Val ROC-AUC: {val_roc_auc:.4f}")
        
        # Early stopping and best model tracking
        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_prauc


def train_model_decoupling(model, train_loader, val_loader, device, df_val, num_class_list,
                          epochs=200, lr=0.001, drw_start_epoch=100):
    """
    Two-stage training for Decoupling method
    """
    # Stage 1: Standard training
    print("Stage 1: Training representation...")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_prauc = 0.0
    best_model_state = None
    
    for epoch in range(drw_start_epoch):
        model.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = loss_fn(outputs, target)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_probabilities = []
        val_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                outputs = model(data)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                val_probabilities.extend(probs.cpu().numpy())
                val_labels.extend(target.numpy())
        
        val_prauc = average_precision_score(val_labels, val_probabilities)
        
        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_model_state = model.state_dict().copy()
        
        if epoch % 20 == 0:
            print(f"Stage 1 - Epoch {epoch+1}/{drw_start_epoch}, "
                  f"Loss: {train_loss/len(train_loader):.4f}, Val PRAUC: {val_prauc:.4f}")
    
    # Load best model from stage 1
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Stage 2: Re-weight training
    print("Stage 2: Re-weighted training...")
    
    # Create weighted sampler for stage 2
    train_labels = []
    for _, target in train_loader:
        train_labels.extend(target.numpy())
    train_labels = np.array(train_labels)
    
    weighted_sampler = create_weighted_sampler(train_labels, "balanced")
    
    # Recreate train dataset and loader for stage 2
    train_dataset = train_loader.dataset
    weighted_train_loader = DataLoader(
        train_dataset, batch_size=train_loader.batch_size, 
        sampler=weighted_sampler
    )
    
    # Continue training with reweighting
    optimizer = optim.Adam(model.parameters(), lr=lr*0.1, weight_decay=1e-4)  # Lower LR
    
    for epoch in range(epochs - drw_start_epoch):
        model.train()
        train_loss = 0.0
        
        for data, target in weighted_train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = loss_fn(outputs, target)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_probabilities = []
        val_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                outputs = model(data)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                val_probabilities.extend(probs.cpu().numpy())
                val_labels.extend(target.numpy())
        
        val_prauc = average_precision_score(val_labels, val_probabilities)
        
        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_model_state = model.state_dict().copy()
        
        if epoch % 10 == 0:
            print(f"Stage 2 - Epoch {epoch+1}/{epochs-drw_start_epoch}, "
                  f"Loss: {train_loss/len(weighted_train_loader):.4f}, Val PRAUC: {val_prauc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_prauc


def train_model_bbn(model, train_loader, val_loader, device, df_val, num_class_list,
                   epochs=200, lr=0.001):
    """
    Training function for BBN (Bilateral-Branch Network)
    """
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    best_val_prauc = 0.0
    best_model_state = None
    patience = 30
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # BBN forward pass returns tuple of (conv_out, rebal_out)
            conv_out, rebal_out = model(data, branch="both")
            
            # Calculate losses for both branches
            conv_loss = nn.CrossEntropyLoss()(conv_out, target)
            rebal_loss = nn.CrossEntropyLoss()(rebal_out, target)
            
            # Combined loss
            total_loss = conv_loss + rebal_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_train_loss = train_loss / num_batches
        
        # Validation
        model.eval()
        val_probabilities = []
        val_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                
                # Get ensemble prediction
                conv_out, rebal_out = model(data, branch="both")
                conv_probs = torch.softmax(conv_out, dim=1)[:, 1]
                rebal_probs = torch.softmax(rebal_out, dim=1)[:, 1]
                
                # Ensemble probabilities
                ensemble_probs = 0.5 * conv_probs + 0.5 * rebal_probs
                
                val_probabilities.extend(ensemble_probs.cpu().numpy())
                val_labels.extend(target.numpy())
        
        val_probabilities = np.array(val_probabilities)
        val_labels = np.array(val_labels)
        
        val_prauc = average_precision_score(val_labels, val_probabilities)
        val_roc_auc = roc_auc_score(val_labels, val_probabilities)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val PRAUC: {val_prauc:.4f}, Val ROC-AUC: {val_roc_auc:.4f}")
        
        # Early stopping
        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_prauc


def train_and_evaluate_combination(fps_combination, df_train, df_val, args, parent_dir):
    """Train and evaluate a single fingerprint combination."""
    print(f"\n{'='*60}")
    print(f"Evaluating combination: {','.join(fps_combination)} with {args.imbalanced}")
    print(f"{'='*60}")
    
    # Create combination string for file naming
    combo_str = "_".join(fps_combination)
    
    # Create log directory for this combination
    combo_log_dir = os.path.join(args.log_dir, f"combo_{combo_str}_{args.imbalanced}")
    os.makedirs(combo_log_dir, exist_ok=True)
    
    try:
        # Process data with concatenated fingerprints
        print("Processing fingerprints...")
        TrainData = concatenate_fingerprints(df_train, fps_combination)
        ValData = concatenate_fingerprints(df_val, fps_combination)
        TrainLabel = df_train["LABEL"].values
        DevLabel = df_val["LABEL"].values
        
        print(f"Training data shape: {TrainData.shape}")
        print(f"Validation data shape: {ValData.shape}")
        
        # Calculate class distribution for loss function initialization
        num_class_list = [np.sum(TrainLabel == 0), np.sum(TrainLabel == 1)]
        print(f"Class distribution: {num_class_list}")

        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(TrainData)
        y_train = torch.LongTensor(TrainLabel)
        X_val = torch.FloatTensor(ValData)
        y_val = torch.LongTensor(DevLabel)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # Initialize model based on method with adaptive hidden dimensions
        input_dim = TrainData.shape[1]
        hidden_dims = get_adaptive_hidden_dims(input_dim)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Input dimension: {input_dim}")
        print(f"Adaptive hidden dimensions: {hidden_dims}")
        
        if args.imbalanced == "BBN":
            model = BBNModel(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.3, num_classes=2).to(device)
        else:
            model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.3).to(device)

        print(f"Training with {args.imbalanced} method...")
        
        # Train model based on method
        if args.imbalanced == "DECOUPLING":
            model, best_val_prauc = train_model_decoupling(
                model, train_loader, val_loader, device, df_val, num_class_list,
                epochs=200, lr=0.001, drw_start_epoch=100
            )
        elif args.imbalanced == "BBN":
            model, best_val_prauc = train_model_bbn(
                model, train_loader, val_loader, device, df_val, num_class_list,
                epochs=200, lr=0.001
            )
        else:
            # All other methods use the standard training loop
            loss_fn = get_loss_function(args.imbalanced, num_class_list, device)
            model, best_val_prauc = train_model(
                model, train_loader, val_loader, loss_fn, device, df_val,
                method_type=args.imbalanced, num_class_list=num_class_list,
                epochs=200, lr=0.001
            )

        # Save the model
        best_model_path = os.path.join(combo_log_dir, "best_model.pth")
        model_config = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,  # Save adaptive dimensions
            'dropout': 0.3
        }
        if args.imbalanced == "BBN":
            model_config['num_classes'] = 2
            
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'imbalanced_method': args.imbalanced,
            'fps_combination': fps_combination,
            'model_type': 'BBN' if args.imbalanced == "BBN" else 'MLP'
        }, best_model_path)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            X_val_device = X_val.to(device)
            
            if args.imbalanced == "BBN":
                # Use ensemble prediction for BBN
                model_output = model(X_val_device, branch="both")
                if isinstance(model_output, tuple) and len(model_output) == 2:
                    conv_output, rebal_output = model_output
                else:
                    # Fallback: get outputs separately
                    conv_output = model(X_val_device, branch="conv")
                    rebal_output = model(X_val_device, branch="rebal")
                    if isinstance(conv_output, tuple):
                        conv_output = conv_output[0]
                    if isinstance(rebal_output, tuple):
                        rebal_output = rebal_output[0]
                
                conv_probs = torch.softmax(conv_output, dim=1)[:, 1]
                rebal_probs = torch.softmax(rebal_output, dim=1)[:, 1]
                probabilities = (0.5 * conv_probs + 0.5 * rebal_probs).cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
            else:
                outputs = model(X_val_device)
                probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        # Calculate metrics
        prauc = average_precision_score(DevLabel, probabilities)
        roc_auc = roc_auc_score(DevLabel, probabilities)
        
        print(f"Final PRAUC on validation set: {prauc:.4f}")
        print(f"Final ROC-AUC on validation set: {roc_auc:.4f}")

        # Calculate cluster-based metrics
        n_hits_50, clusters_50, cluster_prauc_50 = calculate_cluster_metrics(df_val, probabilities, 50)
        n_hits_200, clusters_200, cluster_prauc_200 = calculate_cluster_metrics(df_val, probabilities, 200)
        n_hits_500, clusters_500, cluster_prauc_500 = calculate_cluster_metrics(df_val, probabilities, 500)
        
        all_clusters = df_val[df_val["LABEL"] == 1].drop_duplicates("CLUSTER_LABEL").shape[0]
        print(f"All positive clusters in validation set: {all_clusters}")
        print(f"Top 50 selection - Hits: {n_hits_50}, Unique clusters: {clusters_50}, Cluster PRAUC: {cluster_prauc_50:.4f}")
        print(f"Top 200 selection - Hits: {n_hits_200}, Unique clusters: {clusters_200}, Cluster PRAUC: {cluster_prauc_200:.4f}")
        print(f"Top 500 selection - Hits: {n_hits_500}, Unique clusters: {clusters_500}, Cluster PRAUC: {cluster_prauc_500:.4f}")

        # Update results CSV
        combo_name = ",".join(fps_combination)
        model_name = f"BBN_{args.imbalanced}" if args.imbalanced == "BBN" else f"MLP_{args.imbalanced}"
        update_results_csv(
            parent_dir, model_name, combo_name, prauc, roc_auc, args.imbalanced,
            n_hits_50, clusters_50, cluster_prauc_50,
            n_hits_200, clusters_200, cluster_prauc_200,
            n_hits_500, clusters_500, cluster_prauc_500
        )

        # Save predictions
        df_predictions_val = pd.DataFrame({
            "SMILES": df_val["SMILES"],
            "PredictedScore": probabilities,
            "PredictedLabel": predictions,
        })
        predictions_path = os.path.join(combo_log_dir, "val_predictions.csv")
        df_predictions_val.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")
        
        return {
            'combination': combo_name,
            'prauc': prauc,
            'roc_auc': roc_auc,
            'hits_50': n_hits_50,
            'clusters_50': clusters_50,
            'cluster_prauc_50': cluster_prauc_50
        }
        
    except Exception as e:
        print(f"Error processing combination {','.join(fps_combination)}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main(args):
    # Setup
    os.makedirs(args.log_dir, exist_ok=True)
    parent_dir = os.path.dirname(os.path.abspath(args.log_dir))
    
    # Load datasets
    df_train = Dataset("../../datasets/DREAM/Train_Dataset_DREAM.parquet").get_dataframe()
    if 'Label' in df_train.columns and 'LABEL' not in df_train.columns:
        df_train.rename(columns={'Label': 'LABEL'}, inplace=True)
    elif 'LABEL' not in df_train.columns and 'Label' not in df_train.columns:
        raise ValueError("Neither 'LABEL' nor 'Label' column found in training data")
    df_val = Dataset("../../datasets/DREAM/Val_Dataset_DREAM_Step2.csv").get_dataframe()

    print("------------------------------------------------------------")
    print("Number of binders in the training set:", (df_train["LABEL"] == 1).sum())
    print("Number of non-binders in the training set:", (df_train["LABEL"] == 0).sum())
    print("------------------------------------------------------------")
    print("Number of binders in the validation set:", (df_val["LABEL"] == 1).sum())
    print("Number of non-binders in the validation set:", (df_val["LABEL"] == 0).sum())
    print("------------------------------------------------------------")
    
    # Check if this is a combination run
    if "," in args.fps_type:
        # This is a combination run - evaluate ONLY the exact combination specified
        fps_types = [fp.strip() for fp in args.fps_type.split(',')]
        
        # Validate fingerprint types
        valid_fps = ["MACCS", "RDK", "AVALON", "ATOMPAIR"]
        for fp in fps_types:
            if fp not in valid_fps:
                raise ValueError(f"Unsupported fingerprint type: {fp}. Valid types: {valid_fps}")
        
        print(f"Evaluating exact fingerprint combination: {fps_types} with {args.imbalanced}")
        
        # Evaluate only the specified combination
        result = train_and_evaluate_combination(tuple(fps_types), df_train, df_val, args, parent_dir)
        
        if result:
            print(f"\n{'='*80}")
            print(f"RESULT FOR FINGERPRINT COMBINATION: {result['combination']} WITH {args.imbalanced}")
            print(f"{'='*80}")
            print(f"PRAUC: {result['prauc']:.4f}")
            print(f"ROC-AUC: {result['roc_auc']:.4f}")
            print(f"Top 50 - Hits: {result['hits_50']}, Clusters: {result['clusters_50']}")
        else:
            print("Combination evaluation failed.")
    
    else:
        # Single fingerprint type - use original logic
        selected_fps = args.fps_type
        if selected_fps not in ["MACCS", "RDK", "AVALON", "ATOMPAIR"]:
            raise ValueError(f"Unsupported fingerprint type: {selected_fps}")

        print(f"Single fingerprint processing with {args.imbalanced}: {selected_fps}")
        
        # Process data
        TrainData = ProcessData(df_train, selected_fps).get_data()
        ValData = ProcessData(df_val, selected_fps).get_data()
        TrainLabel = df_train["LABEL"].values
        DevLabel = df_val["LABEL"].values
        
        print("TrainData shape:", TrainData.shape)
        print("ValData shape:", ValData.shape)

        # Calculate class distribution for loss function initialization
        num_class_list = [np.sum(TrainLabel == 0), np.sum(TrainLabel == 1)]
        print(f"Class distribution: {num_class_list}")

        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(TrainData)
        y_train = torch.LongTensor(TrainLabel)
        X_val = torch.FloatTensor(ValData)
        y_val = torch.LongTensor(DevLabel)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # Initialize model based on method with adaptive hidden dimensions
        input_dim = TrainData.shape[1]
        hidden_dims = get_adaptive_hidden_dims(input_dim)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Input dimension: {input_dim}")
        print(f"Adaptive hidden dimensions: {hidden_dims}")
        
        if args.imbalanced == "BBN":
            model = BBNModel(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.3, num_classes=2).to(device)
        else:
            model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=0.3).to(device)

        print(f"Training with {args.imbalanced} method...")
        
        # Train model based on method
        if args.imbalanced == "DECOUPLING":
            model, best_val_prauc = train_model_decoupling(
                model, train_loader, val_loader, device, df_val, num_class_list,
                epochs=200, lr=0.001, drw_start_epoch=100
            )
        elif args.imbalanced == "BBN":
            model, best_val_prauc = train_model_bbn(
                model, train_loader, val_loader, device, df_val, num_class_list,
                epochs=200, lr=0.001
            )
        else:
            # All other methods use the standard training loop
            loss_fn = get_loss_function(args.imbalanced, num_class_list, device)
            model, best_val_prauc = train_model(
                model, train_loader, val_loader, loss_fn, device, df_val,
                method_type=args.imbalanced, num_class_list=num_class_list,
                epochs=200, lr=0.001
            )

        # Save the model
        best_model_path = os.path.join(args.log_dir, "best_model.pth")
        model_config = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,  # Use adaptive dimensions
            'dropout': 0.3
        }
        if args.imbalanced == "BBN":
            model_config['num_classes'] = 2
            
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config,
            'imbalanced_method': args.imbalanced,
            'fps_type': args.fps_type,
            'model_type': 'BBN' if args.imbalanced == "BBN" else 'MLP'
        }, best_model_path)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            X_val_device = X_val.to(device)
            
            if args.imbalanced == "BBN":
                # Use ensemble prediction for BBN
                model_output = model(X_val_device, branch="both")
                if isinstance(model_output, tuple) and len(model_output) == 2:
                    conv_output, rebal_output = model_output
                else:
                    # Fallback: get outputs separately
                    conv_output = model(X_val_device, branch="conv")
                    rebal_output = model(X_val_device, branch="rebal")
                    if isinstance(conv_output, tuple):
                        conv_output = conv_output[0]
                    if isinstance(rebal_output, tuple):
                        rebal_output = rebal_output[0]
                
                conv_probs = torch.softmax(conv_output, dim=1)[:, 1]
                rebal_probs = torch.softmax(rebal_output, dim=1)[:, 1]
                probabilities = (0.5 * conv_probs + 0.5 * rebal_probs).cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
            else:
                outputs = model(X_val_device)
                probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        # Calculate metrics
        prauc = average_precision_score(DevLabel, probabilities)
        roc_auc = roc_auc_score(DevLabel, probabilities)
        
        print(f"Final PRAUC on validation set: {prauc:.4f}")
        print(f"Final ROC-AUC on validation set: {roc_auc:.4f}")

        # Calculate cluster-based metrics
        n_hits_50, clusters_50, cluster_prauc_50 = calculate_cluster_metrics(df_val, probabilities, 50)
        n_hits_200, clusters_200, cluster_prauc_200 = calculate_cluster_metrics(df_val, probabilities, 200)
        n_hits_500, clusters_500, cluster_prauc_500 = calculate_cluster_metrics(df_val, probabilities, 500)
        
        all_clusters = df_val[df_val["LABEL"] == 1].drop_duplicates("CLUSTER_LABEL").shape[0]
        print(f"All positive clusters in validation set: {all_clusters}")
        print(f"Top 50 selection - Hits: {n_hits_50}, Unique clusters: {clusters_50}, Cluster PRAUC: {cluster_prauc_50:.4f}")
        print(f"Top 200 selection - Hits: {n_hits_200}, Unique clusters: {clusters_200}, Cluster PRAUC: {cluster_prauc_200:.4f}")
        print(f"Top 500 selection - Hits: {n_hits_500}, Unique clusters: {clusters_500}, Cluster PRAUC: {cluster_prauc_500:.4f}")

        # Update results CSV
        model_name = f"BBN_{args.imbalanced}" if args.imbalanced == "BBN" else f"MLP_{args.imbalanced}"
        update_results_csv(
            parent_dir, model_name, args.fps_type, prauc, roc_auc, args.imbalanced,
            n_hits_50, clusters_50, cluster_prauc_50,
            n_hits_200, clusters_200, cluster_prauc_200,
            n_hits_500, clusters_500, cluster_prauc_500
        )

        # Save predictions
        df_predictions_val = pd.DataFrame({
            "SMILES": df_val["SMILES"],
            "PredictedScore": probabilities,
            "PredictedLabel": predictions,
        })
        predictions_path = os.path.join(args.log_dir, "val_predictions.csv")
        df_predictions_val.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MLP with imbalanced learning methods for DREAM Challenge 2025"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory to save logs and model checkpoints",
    )
    parser.add_argument(
        "--fps_type",
        type=str,
        required=True,
        help="Fingerprint type: single ['MACCS', 'RDK', 'AVALON', 'ATOMPAIR'] or comma-separated combinations (e.g., 'MACCS,RDK,AVALON,ATOMPAIR')",
    )
    parser.add_argument(
        "--imbalanced",
        type=str,
        required=True,
        choices=["CB_F", "BS", "CB_CE", "CS", "IB", "CDT", "CE", "MIXUP", "REMIX", "DECOUPLING", "BBN"],
        help="Imbalanced learning method: 'CB_F' (Class-balanced focal loss), "
        "'BS' (Balanced softmax), 'CB_CE' (Class-balanced cross-entropy loss), "
        "'CS' (Cost-sensitive cross-entropy loss), 'IB' (Influence-balanced loss), "
        "'CDT' (Class-dependent temperatures), 'CE' (Standard cross-entropy loss), "
        "'MIXUP' (Info augmentation with mixup), 'REMIX' (Info augmentation with remix), "
        "'DECOUPLING' (Decoupling representation and classifier), 'BBN' (Bilateral-branch network)",
    )
    args = parser.parse_args()

    main(args)