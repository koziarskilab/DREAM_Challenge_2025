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

# Import the loss functions from the molecular_imbalanced_benchmark
from loss import (
    ClassBalanceFocal,
    BalancedSoftmaxCE,
    ClassBalanceCE,
    CostSensitiveCE,
    InfluenceBalancedLoss,
    CDT,  
)


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
                    "ALPHA": 1000.0,
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


def train_model_decoupling(model, train_loader, val_loader, device, df_val, num_class_list, 
                          epochs=100, lr=0.001, drw_start_epoch=80):
    """Train model with Decoupling strategy (representation learning + classifier re-training)"""
    
    # Phase 1: Representation learning with weighted sampling
    print("Phase 1: Representation learning with weighted sampling...")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    best_prauc = -1
    best_model_features = None
    patience_counter = 0
    patience = 20
    
    # Create weighted sampler for representation learning
    train_labels = []
    for _, batch_y in train_loader:
        train_labels.extend(batch_y.numpy())
    train_labels = np.array(train_labels)
    
    weighted_sampler = create_weighted_sampler(train_labels, "balanced")
    weighted_train_loader = DataLoader(
        train_loader.dataset, 
        batch_size=train_loader.batch_size, 
        sampler=weighted_sampler
    )
    
    representation_epochs = drw_start_epoch
    epoch_pbar = tqdm(range(representation_epochs), desc="Representation Learning", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        
        batch_pbar = tqdm(weighted_train_loader, desc=f"Epoch {epoch+1}/{representation_epochs}", leave=False, unit="batch")
        
        for batch_x, batch_y in batch_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            batch_pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(weighted_train_loader)
        
        # Validation
        model.eval()
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_probs)
        val_prauc = average_precision_score(val_labels, val_probs)
        
        scheduler.step(val_prauc)
        
        if val_prauc > best_prauc:
            best_prauc = val_prauc
            best_model_features = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        epoch_pbar.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Val AUC": f"{val_auc:.4f}",
            "Val PRAUC": f"{val_prauc:.4f}",
            "Best PRAUC": f"{best_prauc:.4f}",
            "Patience": f"{patience_counter}/{patience}"
        })
        
        if patience_counter >= patience:
            break
    
    epoch_pbar.close()
    
    # Load best representation
    if best_model_features is not None:
        model.load_state_dict(best_model_features)
    
    # Phase 2: Classifier re-training with balanced sampling
    print("Phase 2: Classifier re-training with balanced sampling...")
    
    # Freeze feature layers (all except the last layer)
    for name, param in model.named_parameters():
        if 'model.6' not in name:  # Don't freeze the final linear layer
            param.requires_grad = False
    
    # Re-initialize optimizer for only the classifier layer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr*0.1, weight_decay=1e-4)  # Lower learning rate
    
    # Use balanced sampler for classifier re-training
    balanced_sampler = create_weighted_sampler(train_labels, "balanced")
    balanced_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=balanced_sampler
    )
    
    classifier_epochs = epochs - drw_start_epoch
    best_prauc_phase2 = -1
    best_model_final = None
    patience_counter = 0
    
    epoch_pbar = tqdm(range(classifier_epochs), desc="Classifier Re-training", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        
        batch_pbar = tqdm(balanced_train_loader, desc=f"Epoch {epoch+1}/{classifier_epochs}", leave=False, unit="batch")
        
        for batch_x, batch_y in batch_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            batch_pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(balanced_train_loader)
        
        # Validation
        model.eval()
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_probs)
        val_prauc = average_precision_score(val_labels, val_probs)
        
        if val_prauc > best_prauc_phase2:
            best_prauc_phase2 = val_prauc
            best_model_final = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        epoch_pbar.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Val AUC": f"{val_auc:.4f}",
            "Val PRAUC": f"{val_prauc:.4f}",
            "Best PRAUC": f"{best_prauc_phase2:.4f}",
            "Patience": f"{patience_counter}/{patience}"
        })
        
        if patience_counter >= patience:
            break
    
    epoch_pbar.close()
    
    # Load best final model
    if best_model_final is not None:
        model.load_state_dict(best_model_final)
        best_prauc = best_prauc_phase2
    
    return model, best_prauc


def train_model_bbn(model, train_loader, val_loader, device, df_val, num_class_list, 
                   epochs=100, lr=0.001):
    """Train BBN model with bilateral branches (without mixup)"""
    
    # Create different samplers
    train_labels = []
    for _, batch_y in train_loader:
        train_labels.extend(batch_y.numpy())
    train_labels = np.array(train_labels)
    
    # Reverse sampler (for re-balancing branch) - emphasizes minority class
    class_counts = np.bincount(train_labels)
    # Reverse the frequency: give more weight to minority class
    reverse_weights = class_counts.max() / class_counts
    sample_weights = reverse_weights[train_labels]
    reverse_sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)
    
    reverse_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=reverse_sampler
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()
    
    best_prauc = -1
    best_model_state = None
    patience_counter = 0
    patience = 20
    
    epoch_pbar = tqdm(range(epochs), desc="BBN Training", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        batch_count = 0
        
        # Create iterators for both samplers
        uniform_iter = iter(train_loader)
        reverse_iter = iter(reverse_train_loader)
        
        # Train with both samplers
        max_batches = min(len(train_loader), len(reverse_train_loader))
        
        batch_pbar = tqdm(range(max_batches), desc=f"Epoch {epoch+1}/{epochs}", leave=False, unit="batch")
        
        for batch_idx in batch_pbar:
            try:
                # Get batch from uniform sampler (conventional branch)
                batch_x_uniform, batch_y_uniform = next(uniform_iter)
                batch_x_uniform, batch_y_uniform = batch_x_uniform.to(device), batch_y_uniform.to(device)
                
                # Get batch from reverse sampler (re-balancing branch)
                batch_x_reverse, batch_y_reverse = next(reverse_iter)
                batch_x_reverse, batch_y_reverse = batch_x_reverse.to(device), batch_y_reverse.to(device)
                
            except StopIteration:
                break
            
            optimizer.zero_grad()
            
            # Train both branches separately (no mixup)
            # Forward pass for conventional branch with uniform sampling
            conv_output = model(batch_x_uniform, branch="conv")
            # Handle case where model returns tuple
            if isinstance(conv_output, tuple):
                conv_output = conv_output[0]
            conv_loss = criterion(conv_output, batch_y_uniform)
            
            # Forward pass for re-balancing branch with reverse sampling
            rebal_output = model(batch_x_reverse, branch="rebal")
            # Handle case where model returns tuple
            if isinstance(rebal_output, tuple):
                rebal_output = rebal_output[0]
            rebal_loss = criterion(rebal_output, batch_y_reverse)
            
            # Combine losses (equal weighting)
            total_loss = 0.5 * conv_loss + 0.5 * rebal_loss
            
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
            batch_count += 1
            
            batch_pbar.set_postfix({"Batch Loss": f"{total_loss.item():.4f}"})
        
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        
        # Validation (use ensemble of both branches)
        model.eval()
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # Get outputs from both branches
                model_output = model(batch_x, branch="both")
                if isinstance(model_output, tuple) and len(model_output) == 2:
                    conv_output, rebal_output = model_output
                else:
                    # Fallback: get outputs separately
                    conv_output = model(batch_x, branch="conv")
                    rebal_output = model(batch_x, branch="rebal")
                    if isinstance(conv_output, tuple):
                        conv_output = conv_output[0]
                    if isinstance(rebal_output, tuple):
                        rebal_output = rebal_output[0]
                
                # Ensemble prediction
                conv_probs = torch.softmax(conv_output, dim=1)[:, 1]
                rebal_probs = torch.softmax(rebal_output, dim=1)[:, 1]
                ensemble_probs = 0.5 * conv_probs + 0.5 * rebal_probs
                
                val_probs.extend(ensemble_probs.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_probs)
        val_prauc = average_precision_score(val_labels, val_probs)
        
        scheduler.step(val_prauc)
        
        if val_prauc > best_prauc:
            best_prauc = val_prauc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        epoch_pbar.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Val AUC": f"{val_auc:.4f}",
            "Val PRAUC": f"{val_prauc:.4f}",
            "Best PRAUC": f"{best_prauc:.4f}",
            "Patience": f"{patience_counter}/{patience}"
        })
        
        if patience_counter >= patience:
            epoch_pbar.set_description(f"Early stopping at epoch {epoch+1}")
            break
    
    epoch_pbar.close()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No improvement found during training, keeping final model state")
        best_prauc = val_prauc
    
    return model, best_prauc


def train_model(model, train_loader, val_loader, loss_fn, device, df_val, method_type="CE", 
                num_class_list=None, epochs=100, lr=0.001):
    """Train the MLP model with the specified method"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    best_prauc = -1
    best_model_state = None
    patience_counter = 0
    patience = 20
    
    # Mixup/Remix parameters
    mixup_alpha = 1.0
    remix_kappa = 3.0
    remix_tau = 0.5
    
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        
        # Update loss function for epoch-dependent methods
        if hasattr(loss_fn, 'update'):
            loss_fn.update(epoch + 1)
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, unit="batch")
        
        for batch_x, batch_y in batch_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            if method_type == "MIXUP":
                # Apply mixup
                mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, mixup_alpha, device)
                outputs = model(mixed_x)
                loss = mixup_criterion(loss_fn, outputs, y_a, y_b, lam)
                
            elif method_type == "REMIX":
                # Apply remix
                mixed_x, y_a, y_b, lam = remix_data(
                    batch_x, batch_y, mixup_alpha, remix_kappa, remix_tau, num_class_list, device
                )
                outputs = model(mixed_x)
                loss = mixup_criterion(loss_fn, outputs, y_a, y_b, lam)
                
            else:
                # Standard training or other loss-based methods
                outputs = model(batch_x)
                
                if isinstance(loss_fn, InfluenceBalancedLoss):
                    features = model.forward_features(batch_x)
                    loss = loss_fn(outputs, batch_y, feature=features)
                else:
                    loss = loss_fn(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            batch_pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase (always standard evaluation)
        model.eval()
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_probs)
        val_prauc = average_precision_score(val_labels, val_probs)
        
        scheduler.step(val_prauc)
        
        if val_prauc > best_prauc:
            best_prauc = val_prauc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        epoch_pbar.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Val AUC": f"{val_auc:.4f}",
            "Val PRAUC": f"{val_prauc:.4f}",
            "Best PRAUC": f"{best_prauc:.4f}",
            "Patience": f"{patience_counter}/{patience}"
        })
        
        if patience_counter >= patience:
            epoch_pbar.set_description(f"Early stopping at epoch {epoch+1}")
            break
    
    epoch_pbar.close()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No improvement found during training, keeping final model state")
        best_prauc = val_prauc
    
    return model, best_prauc


def update_results_csv(parent_dir, model_type, fps_type, prauc, roc_auc, metric,
                       hits_50=None, clusters_50=None, cluster_prauc_50=None,
                       hits_200=None, clusters_200=None, cluster_prauc_200=None,
                       hits_500=None, clusters_500=None, cluster_prauc_500=None):
    """Update the results CSV file with enhanced metrics"""
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


def main(args):
    # Setup
    os.makedirs(args.log_dir, exist_ok=True)
    parent_dir = os.path.dirname(os.path.abspath(args.log_dir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets
    df_train = Dataset("../datasets/DREAM/Train_Dataset_DREAM.parquet").get_dataframe()
    df_val = Dataset("../datasets/DREAM/Val_Dataset_DREAM_Step2.csv").get_dataframe()

    print("------------------------------------------------------------")
    print("Number of binders in the training set:", (df_train["LABEL"] == 1).sum())
    print("Number of non-binders in the training set:", (df_train["LABEL"] == 0).sum())
    print("------------------------------------------------------------")
    print("Number of binders in the validation set:", (df_val["LABEL"] == 1).sum())
    print("Number of non-binders in the validation set:", (df_val["LABEL"] == 0).sum())
    print("------------------------------------------------------------")

    # Process data
    selected_fps = args.fps_type
    if selected_fps not in ["MACCS", "RDK", "AVALON", "ATOMPAIR"]:
        raise ValueError(f"Unsupported fingerprint type: {selected_fps}")

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

    # Initialize model based on method
    input_dim = TrainData.shape[1]
    
    if args.imbalanced == "BBN":
        model = BBNModel(input_dim=input_dim, hidden_dims=[256, 256, 256], dropout=0.3, num_classes=2).to(device)
    else:
        model = MLP(input_dim=input_dim, hidden_dims=[256, 256, 256], dropout=0.3).to(device)

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
        'hidden_dims': [256, 256, 256],
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
        help="Fingerprint type: ['MACCS', 'RDK', 'AVALON', 'ATOMPAIR']",
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