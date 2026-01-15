import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import os
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc
import pdb  # Add this import for debugging
import datetime  # Add this import for timestamp

# Import from baseline
from helper import Dataset, ProcessData
from dataset import SMILESDataset, collate_fn_gcn, collate_fn_transformer
from models.gcn import GCNClassifier
from models.transformer import TransformerClassifier
from models.mlp import MLPClassifier

# Import imbalanced learning loss functions
from loss import get_loss_function

# Add these imports
from da_module import DataAugmentationCombiner

def update_results_csv(parent_dir, model_type, fps_type, prauc, roc_auc, metric,
                       hits_50=None, clusters_50=None, cluster_prauc_50=None,
                       hits_200=None, clusters_200=None, cluster_prauc_200=None,
                       hits_500=None, clusters_500=None, cluster_prauc_500=None,
                       hits_5000=None, clusters_5000=None, cluster_prauc_5000=None):
    """Update the results CSV file with enhanced metrics including top 5000"""
    # Use combination-specific CSV file name
    if "," in fps_type:
        combination_name = fps_type.replace(",", "_")
        results_file = os.path.join(parent_dir, f"model_results_{combination_name}.csv")
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
    # Add top 5000 metrics
    if hits_5000 is not None:
        new_data["Hits_Top5000"] = hits_5000
    if clusters_5000 is not None:
        new_data["Clusters_Top5000"] = clusters_5000
    if cluster_prauc_5000 is not None:
        new_data["ClusterPRAUC_Top5000"] = cluster_prauc_5000

    if os.path.exists(results_file):
        df_results = pd.read_csv(results_file)
        df_results = pd.concat([df_results, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df_results = pd.DataFrame([new_data])

    df_results.to_csv(results_file, index=False)
    print(f"Results updated in {results_file}")
    

def calculate_cluster_metrics(df_val, probabilities, top_n):
    """Calculate cluster-based metrics for top N compounds"""
    all_clusters = df_val[df_val["Label"] == 1].drop_duplicates("CLUSTER_LABEL").shape[0]
    
    sorted_indices = probabilities.argsort()[::-1]
    selection = df_val.iloc[sorted_indices[:top_n]].copy()
    selection["Score"] = probabilities[sorted_indices[:top_n]]
    
    hits = selection[selection["Label"] == 1]
    n_hits = hits.shape[0]
    
    if n_hits > 0:
        unique_clusters = hits["CLUSTER_LABEL"].nunique()
        cluster_prauc = average_precision_score(
            selection["Label"], selection["Score"]
        )
    else:
        unique_clusters = 0
        cluster_prauc = 0.0
    
    return n_hits, unique_clusters, cluster_prauc

def aggregate_compound_predictions(probabilities, metadata, original_data):
    """
    Aggregate predictions for compounds that have multiple variants (conformers/tautomers)
    
    Args:
        probabilities: List of prediction probabilities
        metadata: List of metadata dictionaries with compound information
        original_data: Original dataset with compound information
    
    Returns:
        aggregated_probabilities: Array of aggregated probabilities per unique compound
        original_indices: Array of original compound indices
    """
    if not metadata or not original_data:
        return np.array(probabilities), np.arange(len(probabilities))
    
    # Group predictions by original compound index
    compound_predictions = {}
    
    for prob, meta in zip(probabilities, metadata):
        original_idx = meta.get('original_idx', meta.get('compound_idx', 0))
        
        if original_idx not in compound_predictions:
            compound_predictions[original_idx] = []
        compound_predictions[original_idx].append(prob)
    
    # Aggregate predictions (using mean)
    aggregated_probs = []
    original_indices = []
    
    for original_idx in sorted(compound_predictions.keys()):
        compound_probs = compound_predictions[original_idx]
        # Use mean aggregation
        aggregated_prob = np.mean(compound_probs)
        aggregated_probs.append(aggregated_prob)
        original_indices.append(original_idx)
    
    return np.array(aggregated_probs), np.array(original_indices)


def get_criterion_and_class_info(train_dataset, loss_type, device):
    """
    Get the appropriate loss function and calculate class distribution
    
    Args:
        train_dataset: Training dataset
        loss_type: Loss function type ('CE', 'CB_F', 'BS', 'CB_CE', 'CS', 'IB', 'CDT')
        device: Device to use
    
    Returns:
        criterion: Loss function
        num_class_list: List of class sample counts
    """
    # Calculate class distribution from training dataset
    train_labels = []
    
    # Extract labels from different dataset types
    if hasattr(train_dataset, '__getitem__'):
        for i in range(len(train_dataset)):
            try:
                sample = train_dataset[i]
                if len(sample) >= 2:
                    # Get label (second element for most datasets)
                    if isinstance(sample[1], torch.Tensor):
                        train_labels.append(sample[1].item())
                    else:
                        train_labels.append(sample[1])
            except Exception as e:
                # Skip problematic samples
                continue
    
    # Fallback: if we couldn't extract labels, assume balanced dataset
    if not train_labels:
        print("Warning: Could not extract labels from training dataset, assuming balanced classes")
        num_class_list = [1000, 1000]  # Default balanced
    else:
        train_labels = np.array(train_labels)
        num_class_list = [np.sum(train_labels == 0), np.sum(train_labels == 1)]
    
    print(f"Class distribution - Class 0: {num_class_list[0]}, Class 1: {num_class_list[1]}")
    
    # Get appropriate loss function
    if loss_type == "CE":
        # Standard cross-entropy loss
        criterion = nn.CrossEntropyLoss()
    else:
        # Imbalanced learning loss functions
        criterion = get_loss_function(loss_type, num_class_list, device)
    
    return criterion, num_class_list

def evaluate_gcn(model, dataloader, criterion, device, is_validation=False, original_data=None):
    """Evaluation function for GCN"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    probabilities = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            # Handle both old format (graph, labels) and new format (graph, labels, metadata)
            if len(batch_data) == 3:
                graph, labels, metadata = batch_data
            else:
                graph, labels = batch_data
                
            if graph is None:
                continue
                
            graph = graph.to(device)
            labels = labels.to(device)
            
            logits = model(graph)
            
            # Handle different loss function types
            if hasattr(criterion, 'forward'):
                # Custom loss functions from loss.py
                if hasattr(model, 'get_features') and criterion.__class__.__name__ == 'InfluenceBalancedLoss':
                    # For IB loss, we need features
                    features = model.get_features(graph) if hasattr(model, 'get_features') else logits
                    loss = criterion(logits, labels, feature=features)
                else:
                    loss = criterion(logits, labels)
            else:
                # Standard PyTorch loss functions
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            pred = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())  # Get positive class probabilities
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    # Calculate AUC metrics
    try:
        auc_score = roc_auc_score(targets, probabilities)
    except ValueError:
        auc_score = 0.0
    
    if is_validation:
        print(f"Validation - Loss: {avg_loss:.6f}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")
    
    return avg_loss, accuracy, precision, recall, f1, auc_score

def evaluate_transformer(model, dataloader, criterion, device, is_validation=False, original_data=None):
    """Evaluation function for Transformer"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    probabilities = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            # Handle both old format (token_ids, attention_mask, labels) and new format (token_ids, attention_mask, labels, metadata)
            if len(batch_data) == 4:
                token_ids, attention_mask, labels, metadata = batch_data
            else:
                token_ids, attention_mask, labels = batch_data
                
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            logits = model(token_ids, attention_mask)
            
            # Handle different loss function types
            if hasattr(criterion, 'forward'):
                # Custom loss functions from loss.py
                if hasattr(model, 'get_features') and criterion.__class__.__name__ == 'InfluenceBalancedLoss':
                    # For IB loss, we need features
                    features = model.get_features(token_ids, attention_mask) if hasattr(model, 'get_features') else logits
                    loss = criterion(logits, labels, feature=features)
                else:
                    loss = criterion(logits, labels)
            else:
                # Standard PyTorch loss functions
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            pred = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            probabilities.extend(probs[:, 1].cpu().numpy())  # Get positive class probabilities
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    # Calculate AUC metrics
    try:
        auc_score = roc_auc_score(targets, probabilities)
    except ValueError:
        auc_score = 0.0
    
    if is_validation:
        print(f"Validation - Loss: {avg_loss:.6f}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")
    
    return avg_loss, accuracy, precision, recall, f1, auc_score



def train_epoch_gcn_with_augmentation(model, dataloader, criterion, optimizer, device, 
                                      combiner=None, augmentation='none', epoch=None):
    """Enhanced training epoch for GCN with data augmentation"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    # Update loss function if it has update method
    if hasattr(criterion, 'update') and epoch is not None:
        criterion.update(epoch)
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Training GCN ({augmentation})")):
        # Handle batch data format
        if len(batch_data) == 3:
            graph, labels, metadata = batch_data
        else:
            graph, labels = batch_data
            
        if graph is None:
            continue
        
        optimizer.zero_grad()
        
        # Apply data augmentation if specified
        if augmentation == 'mixup' and combiner is not None:
            loss, mixed_acc = combiner.mixup(model, criterion, graph, labels, 'gcn')
            # For mixup, we don't have individual predictions, use mixed accuracy
            total_loss += loss.item()
        elif augmentation == 'remix' and combiner is not None:
            loss, mixed_acc = combiner.remix(model, criterion, graph, labels, 'gcn')
            total_loss += loss.item()
        else:
            # Standard training
            graph = graph.to(device)
            labels = labels.to(device)
            
            logits = model(graph)
            
            # Handle different loss function types
            if hasattr(criterion, 'forward'):
                if hasattr(model, 'get_features') and criterion.__class__.__name__ == 'InfluenceBalancedLoss':
                    features = model.get_features(graph) if hasattr(model, 'get_features') else logits
                    loss = criterion(logits, labels, feature=features)
                else:
                    loss = criterion(logits, labels)
            else:
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            predictions.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Print step loss every 100 steps
        if batch_idx % 100 == 0:
            print(f"  Step {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / len(dataloader)
    
    if augmentation in ['mixup', 'remix']:
        # For augmented training, we can't easily compute accuracy from individual predictions
        accuracy = 0.0  # or use a validation pass to get accuracy
        print(f"  Training Loss: {avg_loss:.6f} (using {augmentation})")
    else:
        accuracy = accuracy_score(targets, predictions)
        print(f"  Training Loss: {avg_loss:.6f}, Training Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy

def train_epoch_transformer_with_augmentation(model, dataloader, criterion, optimizer, device, 
                                              combiner=None, augmentation='none', epoch=None):
    """Enhanced training epoch for Transformer with data augmentation"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    # Update loss function if it has update method
    if hasattr(criterion, 'update') and epoch is not None:
        criterion.update(epoch)
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Training Transformer ({augmentation})")):
        # Handle batch data format
        if len(batch_data) == 4:
            token_ids, attention_mask, labels, metadata = batch_data
        else:
            token_ids, attention_mask, labels = batch_data
        
        optimizer.zero_grad()
        
        # Apply data augmentation if specified
        if augmentation == 'mixup' and combiner is not None:
            loss, mixed_acc = combiner.mixup(model, criterion, (token_ids, attention_mask), labels, 'transformer')
            total_loss += loss.item()
        elif augmentation == 'remix' and combiner is not None:
            loss, mixed_acc = combiner.remix(model, criterion, (token_ids, attention_mask), labels, 'transformer')
            total_loss += loss.item()
        else:
            # Standard training
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            logits = model(token_ids, attention_mask)
            
            # Handle different loss function types
            if hasattr(criterion, 'forward'):
                if hasattr(model, 'get_features') and criterion.__class__.__name__ == 'InfluenceBalancedLoss':
                    features = model.get_features(token_ids, attention_mask) if hasattr(model, 'get_features') else logits
                    loss = criterion(logits, labels, feature=features)
                else:
                    loss = criterion(logits, labels)
            else:
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            predictions.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Print step loss every 100 steps
        if batch_idx % 100 == 0:
            print(f"  Step {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / len(dataloader)
    
    if augmentation in ['mixup', 'remix']:
        accuracy = 0.0
        print(f"  Training Loss: {avg_loss:.6f} (using {augmentation})")
    else:
        accuracy = accuracy_score(targets, predictions)
        print(f"  Training Loss: {avg_loss:.6f}, Training Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy

def train_epoch_mlp_with_augmentation(model, dataloader, criterion, optimizer, device, 
                                      combiner=None, augmentation='none', epoch=None):
    """Enhanced training epoch for MLP with data augmentation"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    # Update loss function if it has update method
    if hasattr(criterion, 'update') and epoch is not None:
        criterion.update(epoch)
    
    for batch_idx, (fingerprints, labels) in enumerate(tqdm(dataloader, desc=f"Training MLP ({augmentation})")):
        optimizer.zero_grad()
        
        # Apply data augmentation if specified
        if augmentation == 'mixup' and combiner is not None:
            loss, mixed_acc = combiner.mixup(model, criterion, fingerprints, labels, 'mlp')
            total_loss += loss.item()
        elif augmentation == 'remix' and combiner is not None:
            loss, mixed_acc = combiner.remix(model, criterion, fingerprints, labels, 'mlp')
            total_loss += loss.item()
        else:
            # Standard training
            fingerprints = fingerprints.to(device)
            labels = labels.to(device)
            
            logits = model(fingerprints)
            
            # Handle different loss function types
            if hasattr(criterion, 'forward'):
                if hasattr(model, 'get_features') and criterion.__class__.__name__ == 'InfluenceBalancedLoss':
                    features = model.get_features(fingerprints) if hasattr(model, 'get_features') else logits
                    loss = criterion(logits, labels, feature=features)
                else:
                    loss = criterion(logits, labels)
            else:
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            predictions.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Print step loss every 100 steps
        if batch_idx % 100 == 0:
            print(f"  Step {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / len(dataloader)
    
    if augmentation in ['mixup', 'remix']:
        accuracy = 0.0
        print(f"  Training Loss: {avg_loss:.6f} (using {augmentation})")
    else:
        accuracy = accuracy_score(targets, predictions)
        print(f"  Training Loss: {avg_loss:.6f}, Training Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='GCN/Transformer/MLP Binder Classifier')
    parser.add_argument('--train_csv', type=str, required=True, help='Training CSV file')
    parser.add_argument('--val_csv', type=str, required=True, help='Validation CSV file')
    parser.add_argument('--smiles_col', type=str, default='SMILES', help='SMILES column name')
    parser.add_argument('--label_col', type=str, default='Label', help='Label column name')
    parser.add_argument('--model_type', type=str, default='gcn', choices=['gcn', 'transformer', 'mlp'], 
                        help='Model type: gcn, transformer, or mlp')
    
    # Add loss function argument
    parser.add_argument('--loss_type', type=str, default='CE', 
                        choices=['CE', 'CB_F', 'BS', 'CB_CE', 'CS', 'IB', 'CDT'],
                        help='Loss function type: CE (Cross-Entropy), CB_F (Class-Balanced Focal), '
                             'BS (Balanced Softmax), CB_CE (Class-Balanced CE), CS (Cost-Sensitive), '
                             'IB (Influence-Balanced), CDT (Class-Dependent Temperatures)')
    
    # Add skip training flag
    parser.add_argument('--skip_training', action='store_true', 
                        help='Skip training and only run validation with randomly initialized model')
    
    # Add patience parameter for early stopping
    parser.add_argument('--patience', type=int, default=10, 
                        help='Number of epochs to wait before early stopping if no improvement')
    
    # Add data augmentation arguments
    parser.add_argument('--augmentation', type=str, default='none', 
                        choices=['none', 'mixup', 'remix'],
                        help='Data augmentation method: none, mixup, or remix')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Alpha parameter for mixup/remix beta distribution')
    parser.add_argument('--remix_kappa', type=float, default=3.0,
                        help='Kappa parameter for remix (class imbalance sensitivity)')
    parser.add_argument('--remix_tau', type=float, default=0.5,
                        help='Tau parameter for remix (threshold)')
    
    # GCN parameters
    parser.add_argument('--hidden_feats', type=int, nargs='+', default=[64, 64], help='Hidden dimensions for GCN')
    
    # Transformer parameters
    parser.add_argument('--vocab_size', type=int, default=65, help='Vocabulary size for Transformer')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum sequence length for Transformer')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size for Transformer')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers for Transformer')
    parser.add_argument('--intermediate_size', type=int, default=512, help='Intermediate size for Transformer')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='Number of attention heads for Transformer')
    parser.add_argument('--attention_probs_dropout', type=float, default=0.1, help='Attention dropout for Transformer')
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.1, help='Hidden dropout for Transformer')
    
    # MLP parameters
    parser.add_argument('--fp_type', type=str, default='MACCS', choices=['MACCS', 'RDK', 'AVALON', 'ATOMPAIR'],
                        help='Fingerprint type for MLP')
    parser.add_argument('--mlp_hidden_dims', type=int, nargs='+', default=[128, 64], help='Hidden dimensions for MLP')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using model: {args.model_type}")
    print(f"Using loss function: {args.loss_type}")
    
    if args.skip_training:
        print("Skip training mode: Using randomly initialized model for validation only")
    
    # Load data for size information and MLP processing
    print("Loading data...")
    
    # Read validation data (always needed)
    if args.val_csv.endswith('.parquet'):
        df_val = pd.read_parquet(args.val_csv)
    else:
        df_val = pd.read_csv(args.val_csv)
    
    print(f"Validation set size: {len(df_val)}")
    
    # Read training data only if not skipping training
    if not args.skip_training:
        if args.train_csv.endswith('.parquet'):
            df_train = pd.read_parquet(args.train_csv)
        else:
            df_train = pd.read_csv(args.train_csv)
        
        print(f"Training set size: {len(df_train)}")
    else:
        df_train = None
        print("Skipping training data loading...")
    
    # Create datasets and dataloaders based on model type
    if args.model_type == 'gcn':
        # Create GCN datasets - pass file paths directly
        if not args.skip_training:
            train_dataset = SMILESDataset(args.train_csv, args.smiles_col, args.label_col, model_type='gcn')
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_gcn)
        else:
            train_dataset = None
            train_loader = None
            
        val_dataset = SMILESDataset(args.val_csv, args.smiles_col, args.label_col, model_type='gcn', is_validation=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_gcn)
        
        # Create GCN model - get node feature size from validation dataset if no training data
        node_feat_size = 74  # Default atom feature size
        try:
            # Try to get a valid sample from train_dataset if available, otherwise val_dataset
            dataset_to_check = train_dataset if train_dataset is not None else val_dataset
            
            # Try to get a valid sample
            for i in range(min(10, len(dataset_to_check))):  # Try first 10 samples
                sample_data = dataset_to_check[i]
                if len(sample_data) == 3:
                    sample_graph, _, _ = sample_data  # graph, label, metadata
                else:
                    sample_graph, _ = sample_data  # graph, label
                    
                # Check if we got a graph or SMILES string
                if isinstance(sample_graph, str):
                    # Convert SMILES to graph
                    try:
                        from dataset import smiles_to_dgl_graph
                        sample_graph = smiles_to_dgl_graph(sample_graph)
                    except Exception as e:
                        print(f"Failed to convert SMILES to graph: {e}")
                        continue
                
                if sample_graph is not None and hasattr(sample_graph, 'ndata'):
                    node_feat_size = sample_graph.ndata['feat'].shape[1]
                    print(f"Detected node feature size: {node_feat_size}")
                    break
                    
            print(f"Using node feature size: {node_feat_size}")
        except Exception as e:
            print(f"Error getting node feature size, using default {node_feat_size}: {e}")
            
        model = GCNClassifier(
            in_feats=node_feat_size,
            hidden_feats=args.hidden_feats,
            n_classes=2,
            dropout=args.dropout
        ).to(device)
        
        train_epoch_fn = lambda model, dataloader, criterion, optimizer, device, epoch: \
            train_epoch_gcn_with_augmentation(model, dataloader, criterion, optimizer, device, 
                                              combiner, args.augmentation, epoch)
        evaluate_fn = evaluate_gcn
        
    elif args.model_type == 'transformer':
        # Create Transformer datasets - pass file paths directly
        if not args.skip_training:
            train_dataset = SMILESDataset(args.train_csv, args.smiles_col, args.label_col, model_type='transformer', max_length=args.max_length)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_transformer)
        else:
            train_dataset = None
            train_loader = None
            
        val_dataset = SMILESDataset(args.val_csv, args.smiles_col, args.label_col, model_type='transformer', max_length=args.max_length, is_validation=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_transformer)
        
        # Create Transformer model with better defaults
        model = TransformerClassifier(
            vocab_size=args.vocab_size,
            hidden_size=max(args.hidden_size, 128),  # Ensure minimum size
            num_layers=max(args.num_layers, 4),      # Ensure minimum depth
            intermediate_size=args.intermediate_size,
            num_attention_heads=args.num_attention_heads,
            attention_probs_dropout=min(args.attention_probs_dropout, 0.1),  # Cap dropout
            hidden_dropout_rate=min(args.hidden_dropout_rate, 0.1),
            num_classes=2
        ).to(device)
        
        train_epoch_fn = lambda model, dataloader, criterion, optimizer, device, epoch: \
            train_epoch_transformer_with_augmentation(model, dataloader, criterion, optimizer, device, 
                                                      combiner, args.augmentation, epoch)
        evaluate_fn = evaluate_transformer
        
    elif args.model_type == 'mlp':
        # For MLP, we need to process fingerprints using the loaded DataFrames
        print("Processing fingerprints...")
        
        # Process fingerprints
        processor = ProcessData()
        
        # Get validation fingerprints (always needed)
        val_smiles = df_val[args.smiles_col].tolist()
        val_fps = processor.get_fingerprints(val_smiles, fp_type=args.fp_type)
        val_labels = df_val[args.label_col].values
        val_dataset = TensorDataset(torch.FloatTensor(val_fps), torch.LongTensor(val_labels))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Get training fingerprints only if not skipping training
        if not args.skip_training:
            train_smiles = df_train[args.smiles_col].tolist()
            train_fps = processor.get_fingerprints(train_smiles, fp_type=args.fp_type)
            train_labels = df_train[args.label_col].values
            train_dataset = TensorDataset(torch.FloatTensor(train_fps), torch.LongTensor(train_labels))
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        else:
            train_dataset = None
            train_loader = None
        
        # Create MLP model
        fp_dim = get_fingerprint_dim(args.fp_type)
        model = MLPClassifier(
            input_dim=fp_dim,
            hidden_dims=args.mlp_hidden_dims,
            num_classes=2,
            dropout=args.dropout
        ).to(device)
        
        train_epoch_fn = lambda model, dataloader, criterion, optimizer, device, epoch: \
            train_epoch_mlp_with_augmentation(model, dataloader, criterion, optimizer, device, 
                                              combiner, args.augmentation, epoch)
        evaluate_fn = None  # MLP uses different evaluation logic
    
    # Get appropriate loss function and class distribution
    if not args.skip_training:
        criterion, num_class_list = get_criterion_and_class_info(train_dataset, args.loss_type, device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    else:
        # For skip training mode, use standard CE loss
        criterion = nn.CrossEntropyLoss()
        num_class_list = [1000, 1000]  # Default balanced
    
    # Initialize data augmentation combiner if needed
    combiner = None
    if args.augmentation in ['mixup', 'remix'] and not args.skip_training:
        combiner = DataAugmentationCombiner(
            device=device,
            num_class_list=num_class_list,
            alpha=args.alpha,
            remix_kappa=args.remix_kappa,
            remix_tau=args.remix_tau
        )
        print(f"Using {args.augmentation} with alpha={args.alpha}")
        if args.augmentation == 'remix':
            print(f"Remix parameters: kappa={args.remix_kappa}, tau={args.remix_tau}")
    
    # Run validation only if skipping training
    if args.skip_training:
        print("Running validation with randomly initialized model...")
        if args.model_type == 'mlp':
            # For MLP, get detailed validation metrics
            model.eval()
            val_probabilities = []
            val_labels = []
            val_predictions = []
            
            with torch.no_grad():
                for fingerprints, labels in val_loader:
                    fingerprints = fingerprints.to(device)
                    
                    logits = model(fingerprints)
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1)
                    
                    val_probabilities.extend(probs[:, 1].cpu().numpy())
                    val_labels.extend(labels.numpy())
                    val_predictions.extend(preds.cpu().numpy())
            
            val_probabilities = np.array(val_probabilities)
            val_labels = np.array(val_labels)
            val_predictions = np.array(val_predictions)
            
            # Calculate metrics
            val_acc = accuracy_score(val_labels, val_predictions)
            val_prec = precision_score(val_labels, val_predictions, average='weighted')
            val_rec = recall_score(val_labels, val_predictions, average='weighted')
            val_f1 = f1_score(val_labels, val_predictions, average='weighted')
            val_prauc = average_precision_score(val_labels, val_probabilities)
            val_roc_auc = roc_auc_score(val_labels, val_probabilities)
            
            # Calculate cluster metrics
            n_hits_50, clusters_50, cluster_prauc_50 = calculate_cluster_metrics(df_val, val_probabilities, 50)
            n_hits_200, clusters_200, cluster_prauc_200 = calculate_cluster_metrics(df_val, val_probabilities, 200)
            n_hits_500, clusters_500, cluster_prauc_500 = calculate_cluster_metrics(df_val, val_probabilities, 500)
            n_hits_5000, clusters_5000, cluster_prauc_5000 = calculate_cluster_metrics(df_val, val_probabilities, 5000)
            
            print(f"Top 50 - Hits: {n_hits_50}, Clusters: {clusters_50}, Cluster PRAUC: {cluster_prauc_50:.4f}")
            print(f"Top 200 - Hits: {n_hits_200}, Clusters: {clusters_200}, Cluster PRAUC: {cluster_prauc_200:.4f}")
            print(f"Top 500 - Hits: {n_hits_500}, Clusters: {clusters_500}, Cluster PRAUC: {cluster_prauc_500:.4f}")
            print(f"Top 5000 - Hits: {n_hits_5000}, Clusters: {clusters_5000}, Cluster PRAUC: {cluster_prauc_5000:.4f}")
            
        else:
            # For GCN/Transformer, use new evaluation with compound SMILES support
            # Check if val_dataset has original_data attribute
            original_data = getattr(val_dataset, 'original_data', None)
            
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate_fn(
                model, val_loader, criterion, device, 
                is_validation=True, original_data=original_data)
            
            # Get probabilities with proper aggregation for cluster metrics
            model.eval()
            all_probabilities = []
            all_metadata = []
            
            with torch.no_grad():
                for batch_data in val_loader:
                    if len(batch_data) == 3 or len(batch_data) == 4:  # Handle both GCN and Transformer formats
                        if args.model_type == 'gcn':
                            if len(batch_data) == 3:
                                graph, labels, batch_metadata = batch_data
                                all_metadata.extend(batch_metadata)
                            else:
                                graph, labels = batch_data
                                # Create dummy metadata for backward compatibility
                                batch_size = labels.size(0)
                                dummy_metadata = [{'original_idx': i, 'compound_variant': 0, 'total_variants': 1} 
                                                for i in range(batch_size)]
                                all_metadata.extend(dummy_metadata)
                            
                            if graph is not None:
                                graph = graph.to(device)
                                logits = model(graph)
                        else:  # transformer
                            if len(batch_data) == 4:
                                token_ids, attention_mask, labels, batch_metadata = batch_data
                                all_metadata.extend(batch_metadata)
                            else:
                                token_ids, attention_mask, labels = batch_data
                                # Create dummy metadata for backward compatibility
                                batch_size = labels.size(0)
                                dummy_metadata = [{'original_idx': i, 'compound_variant': 0, 'total_variants': 1} 
                                                for i in range(batch_size)]
                                all_metadata.extend(dummy_metadata)
                            
                            token_ids = token_ids.to(device)
                            attention_mask = attention_mask.to(device)
                            logits = model(token_ids, attention_mask)
                        
                        probs = torch.softmax(logits, dim=1)
                        all_probabilities.extend(probs[:, 1].cpu().numpy())
            
            # Aggregate probabilities by compound (same as in evaluate functions)
            if original_data:
                val_probabilities, _ = aggregate_compound_predictions(
                    all_probabilities, all_metadata, original_data)
            else:
                val_probabilities = np.array(all_probabilities)

            # Calculate PRAUC and ROC-AUC from aggregated probabilities
            val_labels = df_val[args.label_col].values
            # pdb.set_trace()  # Add debug breakpoint here
            if hasattr(val_dataset, 'original_data') and val_dataset.original_data:
                # Get original labels for aggregated predictions
                val_prauc = average_precision_score(val_labels, val_probabilities)
                val_roc_auc = roc_auc_score(val_labels, val_probabilities)
            else:
                # Fallback to using all predictions (without aggregation)
                val_labels_array = []
                with torch.no_grad():
                    for batch_data in val_loader:
                        if args.model_type == 'gcn':
                            if len(batch_data) >= 2:
                                _, labels = batch_data[0], batch_data[1]
                                val_labels_array.extend(labels.numpy())
                        else:  # transformer
                            if len(batch_data) >= 3:
                                _, _, labels = batch_data[0], batch_data[1], batch_data[2]
                                val_labels_array.extend(labels.numpy())
                
                val_prauc = average_precision_score(val_labels_array, all_probabilities)
                val_roc_auc = roc_auc_score(val_labels_array, all_probabilities)

            print(f"Validation - Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}, PRAUC: {val_prauc:.4f}, ROC-AUC: {val_roc_auc:.4f}")
            
            # Calculate cluster metrics
            n_hits_50, clusters_50, cluster_prauc_50 = calculate_cluster_metrics(df_val, val_probabilities, 50)
            n_hits_200, clusters_200, cluster_prauc_200 = calculate_cluster_metrics(df_val, val_probabilities, 200)
            n_hits_500, clusters_500, cluster_prauc_500 = calculate_cluster_metrics(df_val, val_probabilities, 500)
            n_hits_5000, clusters_5000, cluster_prauc_5000 = calculate_cluster_metrics(df_val, val_probabilities, 5000)
            
            print(f"Top 50 - Hits: {n_hits_50}, Clusters: {clusters_50}, Cluster PRAUC: {cluster_prauc_50:.4f}")
            print(f"Top 200 - Hits: {n_hits_200}, Clusters: {clusters_200}, Cluster PRAUC: {cluster_prauc_200:.4f}")
            print(f"Top 500 - Hits: {n_hits_500}, Clusters: {clusters_500}, Cluster PRAUC: {cluster_prauc_500:.4f}")
            print(f"Top 5000 - Hits: {n_hits_5000}, Clusters: {clusters_5000}, Cluster PRAUC: {cluster_prauc_5000:.4f}")
        
        print("Validation completed!")
        return
    
    # Training loop (only if not skipping training)
    best_val_prauc = 0
    best_val_acc = 0
    best_clusters_5000 = 0  # Change from best_clusters_50 to best_clusters_5000
    patience_counter = 0  # Add patience counter
    
    print("Starting training...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Training
        train_loss, train_acc = train_epoch_fn(model, train_loader, criterion, optimizer, device, epoch + 1)
        print(f"Epoch {epoch+1} Summary - Training Loss: {train_loss:.6f}, Training Accuracy: {train_acc:.4f}")
        
        # Validation
        if args.model_type == 'mlp':
            # For MLP, get detailed validation metrics
            model.eval()
            val_probabilities = []
            val_labels = []
            val_predictions = []
            
            with torch.no_grad():
                for fingerprints, labels in val_loader:
                    fingerprints = fingerprints.to(device)
                    
                    logits = model(fingerprints)
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1)
                    
                    val_probabilities.extend(probs[:, 1].cpu().numpy())
                    val_labels.extend(labels.numpy())
                    val_predictions.extend(preds.cpu().numpy())
            
            val_probabilities = np.array(val_probabilities)
            val_labels = np.array(val_labels)
            val_predictions = np.array(val_predictions)
            
            # Calculate metrics
            val_acc = accuracy_score(val_labels, val_predictions)
            val_prec = precision_score(val_labels, val_predictions, average='weighted')
            val_rec = recall_score(val_labels, val_predictions, average='weighted')
            val_f1 = f1_score(val_labels, val_predictions, average='weighted')
            val_prauc = average_precision_score(val_labels, val_probabilities)
            val_roc_auc = roc_auc_score(val_labels, val_probabilities)
            
            print(f"Validation - Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}, PRAUC: {val_prauc:.4f}, ROC-AUC: {val_roc_auc:.4f}")
            
            # Calculate cluster metrics
            n_hits_50, clusters_50, cluster_prauc_50 = calculate_cluster_metrics(df_val, val_probabilities, 50)
            n_hits_200, clusters_200, cluster_prauc_200 = calculate_cluster_metrics(df_val, val_probabilities, 200)
            n_hits_500, clusters_500, cluster_prauc_500 = calculate_cluster_metrics(df_val, val_probabilities, 500)
            n_hits_5000, clusters_5000, cluster_prauc_5000 = calculate_cluster_metrics(df_val, val_probabilities, 5000)
            
            print(f"Top 50 - Hits: {n_hits_50}, Clusters: {clusters_50}, Cluster PRAUC: {cluster_prauc_50:.4f}")
            print(f"Top 200 - Hits: {n_hits_200}, Clusters: {clusters_200}, Cluster PRAUC: {cluster_prauc_200:.4f}")
            print(f"Top 500 - Hits: {n_hits_500}, Clusters: {clusters_500}, Cluster PRAUC: {cluster_prauc_500:.4f}")
            print(f"Top 5000 - Hits: {n_hits_5000}, Clusters: {clusters_5000}, Cluster PRAUC: {cluster_prauc_5000:.4f}")
            
        else:
            # For GCN/Transformer, use new evaluation with compound SMILES support
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate_fn(
                model, val_loader, criterion, device, 
                is_validation=True, original_data=val_dataset.original_data)
            
            model.eval()
            all_probabilities = []
            all_metadata = []
            
            with torch.no_grad():
                for batch_data in val_loader:
                    if len(batch_data) == 3 or len(batch_data) == 4:  # Handle both GCN and Transformer formats
                        if args.model_type == 'gcn':
                            if len(batch_data) == 3:
                                graph, labels, batch_metadata = batch_data
                                all_metadata.extend(batch_metadata)
                            else:
                                graph, labels = batch_data
                                # Create dummy metadata for backward compatibility
                                batch_size = labels.size(0)
                                dummy_metadata = [{'original_idx': i, 'compound_variant': 0, 'total_variants': 1} 
                                                for i in range(batch_size)]
                                all_metadata.extend(dummy_metadata)
                            
                            if graph is not None:
                                graph = graph.to(device)
                                logits = model(graph)
                        else:  # transformer
                            if len(batch_data) == 4:
                                token_ids, attention_mask, labels, batch_metadata = batch_data
                                all_metadata.extend(batch_metadata)
                            else:
                                token_ids, attention_mask, labels = batch_data
                                # Create dummy metadata for backward compatibility
                                batch_size = labels.size(0)
                                dummy_metadata = [{'original_idx': i, 'compound_variant': 0, 'total_variants': 1} 
                                                for i in range(batch_size)]
                                all_metadata.extend(dummy_metadata)
                            
                            token_ids = token_ids.to(device)
                            attention_mask = attention_mask.to(device)
                            logits = model(token_ids, attention_mask)
                        
                        probs = torch.softmax(logits, dim=1)
                        all_probabilities.extend(probs[:, 1].cpu().numpy())
            
            # Aggregate probabilities by compound (same as in evaluate functions)
            if hasattr(val_dataset, 'original_data') and val_dataset.original_data:
                val_probabilities, orig_indices = aggregate_compound_predictions(
                    all_probabilities, all_metadata, val_dataset.original_data)
                
                val_labels = df_val[args.label_col].values
                
                best_val_prauc = average_precision_score(val_labels, val_probabilities)
                best_val_roc_auc = roc_auc_score(val_labels, val_probabilities)
            else:
                # Fallback to using all predictions
                best_val_prauc = average_precision_score(val_labels, all_probabilities)
                best_val_roc_auc = roc_auc_score(val_labels, all_probabilities)
            
            # Calculate cluster metrics
            n_hits_50, clusters_50, cluster_prauc_50 = calculate_cluster_metrics(df_val, val_probabilities, 50)
            n_hits_200, clusters_200, cluster_prauc_200 = calculate_cluster_metrics(df_val, val_probabilities, 200)
            n_hits_500, clusters_500, cluster_prauc_500 = calculate_cluster_metrics(df_val, val_probabilities, 500)
            n_hits_5000, clusters_5000, cluster_prauc_5000 = calculate_cluster_metrics(df_val, val_probabilities, 5000)
            
            print(f"Top 50 - Hits: {n_hits_50}, Clusters: {clusters_50}, Cluster PRAUC: {cluster_prauc_50:.4f}")
            print(f"Top 200 - Hits: {n_hits_200}, Clusters: {clusters_200}, Cluster PRAUC: {cluster_prauc_200:.4f}")
            print(f"Top 500 - Hits: {n_hits_500}, Clusters: {clusters_500}, Cluster PRAUC: {cluster_prauc_500:.4f}")
            print(f"Top 5000 - Hits: {n_hits_5000}, Clusters: {clusters_5000}, Cluster PRAUC: {cluster_prauc_5000:.4f}")
        
        # Save best model and check for early stopping - use clusters_5000 as metric
        improved = False
        if clusters_5000 > best_clusters_5000:
            best_clusters_5000 = clusters_5000
            
            # Save model with loss type info
            model_save_path = os.path.join(args.save_dir, f'best_{args.model_type}_{args.loss_type}_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': args.model_type,
                'loss_type': args.loss_type,
                'clusters_5000': clusters_5000,
                'epoch': epoch + 1
            }, model_save_path)
            
            print(f"Best model saved! New best clusters_5000: {clusters_5000}")
            improved = True
            
            # Store the best results when improved (but don't update CSV yet)
            if args.model_type == 'mlp':
                best_val_prauc = val_prauc
                best_val_roc_auc = val_roc_auc
                data_type = args.fp_type  # For MLP, use fingerprint type
                # Store best metrics for later CSV update
                best_n_hits_50, best_clusters_50_for_csv, best_cluster_prauc_50 = n_hits_50, clusters_50, cluster_prauc_50
                best_n_hits_200, best_clusters_200, best_cluster_prauc_200 = n_hits_200, clusters_200, cluster_prauc_200
                best_n_hits_500, best_clusters_500, best_cluster_prauc_500 = n_hits_500, clusters_500, cluster_prauc_500
                best_n_hits_5000, best_clusters_5000, best_cluster_prauc_5000 = n_hits_5000, clusters_5000, cluster_prauc_5000
            else:
                # For GCN/Transformer, calculate PRAUC and ROC-AUC from probabilities
                if hasattr(val_dataset, 'original_data') and val_dataset.original_data:
                    # Get aggregated probabilities and targets
                    val_probabilities, orig_indices = aggregate_compound_predictions(
                        all_probabilities, all_metadata, val_dataset.original_data)
                    
                    val_labels = df_val[args.label_col].values
                    
                    best_val_prauc = average_precision_score(val_labels, val_probabilities)
                    best_val_roc_auc = roc_auc_score(val_labels, val_probabilities)
                else:
                    # Fallback to using all predictions
                    best_val_prauc = average_precision_score(val_labels, all_probabilities)
                    best_val_roc_auc = roc_auc_score(val_labels, all_probabilities)
                data_type = "SMILES"  # For GCN/Transformer, use SMILES
                # Store best metrics for later CSV update
                best_n_hits_50, best_clusters_50_for_csv, best_cluster_prauc_50 = n_hits_50, clusters_50, cluster_prauc_50
                best_n_hits_200, best_clusters_200, best_cluster_prauc_200 = n_hits_200, clusters_200, cluster_prauc_200
                best_n_hits_500, best_clusters_500, best_cluster_prauc_500 = n_hits_500, clusters_500, cluster_prauc_500
                best_n_hits_5000, best_clusters_5000, best_cluster_prauc_5000 = n_hits_5000, clusters_5000, cluster_prauc_5000
            
            # Save predictions for best model
            if args.model_type == 'mlp':
                df_predictions_val = pd.DataFrame({
                    "SMILES": df_val[args.smiles_col],
                    "PredictedScore": val_probabilities,
                    "PredictedLabel": val_predictions,
                })
            else:
                # For GCN/Transformer, use original data if available
                if hasattr(val_dataset, 'original_data') and val_dataset.original_data:
                    df_predictions_val = pd.DataFrame({
                        "SMILES": df_val[args.smiles_col],
                        "PredictedScore": val_probabilities,
                        "PredictedLabel": (val_probabilities > 0.5).astype(int),
                    })
                else:
                    # Fallback for when original_data is not available
                    df_predictions_val = pd.DataFrame({
                        "SMILES": df_val[args.smiles_col],
                        "PredictedScore": all_probabilities,
                        "PredictedLabel": (np.array(all_probabilities) > 0.5).astype(int),
                    })
            
            predictions_path = os.path.join(args.save_dir, f"best_{args.model_type}_{args.loss_type}_predictions.csv")
            df_predictions_val.to_csv(predictions_path, index=False)
            print(f"Best predictions saved to {predictions_path}")
        
        # Early stopping logic
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (current clusters_5000: {clusters_5000}, best: {best_clusters_5000})")
        
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
        
        # Update learning rate scheduler if using one
        if 'scheduler' in locals():
            scheduler.step(clusters_5000)  # Use clusters_5000 for scheduler too
    
    print(f"\nTraining completed. Best clusters_5000: {best_clusters_5000}")
    
    # Update results CSV only at the end of training with the best metrics
    parent_dir = os.path.dirname(os.path.abspath(args.save_dir))
    update_results_csv(
        parent_dir, f"{args.model_type.upper()}_{args.loss_type}", 
        data_type,
        best_val_prauc, best_val_roc_auc, "clusters_5000",  # Change metric name
        best_n_hits_50, best_clusters_50_for_csv, best_cluster_prauc_50,
        best_n_hits_200, best_clusters_200, best_cluster_prauc_200,
        best_n_hits_500, best_clusters_500, best_cluster_prauc_500,
        best_n_hits_5000, best_clusters_5000, best_cluster_prauc_5000
    )
    
    # Final summary of best results
    print(f"\n{'='*60}")
    print(f"FINAL BEST RESULTS FOR {args.model_type.upper()} with {args.loss_type}")
    print(f"{'='*60}")
    print(f"Best clusters_5000: {best_clusters_5000}")
    if 'best_val_prauc' in locals():
        print(f"Best PRAUC: {best_val_prauc:.4f}")
        print(f"Best ROC-AUC: {best_val_roc_auc:.4f}")
    print(f"Model saved to: {model_save_path}")
    print(f"Predictions saved to: {os.path.join(args.save_dir, f'best_{args.model_type}_{args.loss_type}_predictions.csv')}")

if __name__ == "__main__":
    main()