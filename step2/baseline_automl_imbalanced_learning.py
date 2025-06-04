from helper import Dataset, ProcessData
import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
)


class MLP(nn.Module):
    """Multi-layer Perceptron for binary classification"""
    def __init__(self, input_dim, hidden_dims=[256, 256, 256], dropout=0.3):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer for binary classification
        layers.append(nn.Linear(prev_dim, 2))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def get_loss_function(loss_type, num_class_list, device):
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
            },
            "train": {
                "two_stage": {
                    "drw": False,
                    "start_epoch": 160,
                }
            }
        }
    }
    
    if loss_type == "CB_F":
        return ClassBalanceFocal(para_dict)
    elif loss_type == "BS":
        return BalancedSoftmaxCE(para_dict)
    elif loss_type == "CB_CE":
        return ClassBalanceCE(para_dict)
    elif loss_type == "CS":
        return CostSensitiveCE(para_dict)
    elif loss_type == "IB":
        return InfluenceBalancedLoss(para_dict)
    elif loss_type == "CE":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_model(model, train_loader, val_loader, loss_fn, device, df_val, epochs=100, lr=0.001):
    """Train the MLP model with the specified loss function"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    best_prauc = -1  # Start with -1 to ensure first epoch saves model
    best_model_state = None
    patience_counter = 0
    patience = 20
    
    # Progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0
        
        # Update loss function for epoch-dependent methods (skip for standard CE)
        if hasattr(loss_fn, 'update'):
            loss_fn.update(epoch + 1)
        
        # Progress bar for batches
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, unit="batch")
        
        for batch_x, batch_y in batch_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # For InfluenceBalancedLoss, we need to pass features
            if isinstance(loss_fn, InfluenceBalancedLoss):
                # Get features from the second-to-last layer
                features = model.model[:-1](batch_x)  # All layers except the last one
                loss = loss_fn(outputs, batch_y, feature=features)
            else:
                loss = loss_fn(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Update batch progress bar with current loss
            batch_pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())
        
        # Calculate validation metrics
        val_auc = roc_auc_score(val_labels, val_probs)
        val_prauc = average_precision_score(val_labels, val_probs)
        
        scheduler.step(val_prauc)
        
        # Early stopping and best model saving based on PRAUC
        if val_prauc > best_prauc:
            best_prauc = val_prauc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Update epoch progress bar with metrics
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
    
    # Load best model (ensure we have a valid state_dict)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No improvement found during training, keeping final model state")
        best_prauc = val_prauc  # Use the last computed value
    
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
    df_val = Dataset("../datasets/DREAM/Val_Dataset_DREAM.csv").get_dataframe()

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

    # Initialize model
    input_dim = TrainData.shape[1]
    model = MLP(input_dim=input_dim, hidden_dims=[256, 256, 256], dropout=0.3).to(device)

    # Initialize loss function
    loss_fn = get_loss_function(args.loss_type, num_class_list, device)
    
    print(f"Training MLP with {args.loss_type} loss...")
    
    # Train model
    model, best_val_prauc = train_model(
        model, train_loader, val_loader, loss_fn, device, df_val,
        epochs=200, lr=0.001
    )

    # Save the model
    best_model_path = os.path.join(args.log_dir, "best_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': input_dim,
            'hidden_dims': [256, 256, 256],
            'dropout': 0.3
        },
        'loss_type': args.loss_type,
        'fps_type': args.fps_type
    }, best_model_path)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        X_val_device = X_val.to(device)
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
    update_results_csv(
        parent_dir, f"MLP_{args.loss_type}", args.fps_type, prauc, roc_auc, args.loss_type,
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
        "--loss_type",
        type=str,
        required=True,
        choices=["CB_F", "BS", "CB_CE", "CS", "IB", "CE"],
        help="Loss function type: 'CB_F' (Class-balanced focal loss), "
        "'BS' (Balanced softmax), 'CB_CE' (Class-balanced cross-entropy loss), "
        "'CS' (Cost-sensitive cross-entropy loss), 'IB' (Influence-balanced loss), "
        "'CE' (Standard cross-entropy loss)",
    )
    args = parser.parse_args()

    main(args)