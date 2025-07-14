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

# Import from baseline
from helper import Dataset, ProcessData
from dataset import SMILESDataset, collate_fn_gcn, collate_fn_transformer
from models.gcn import GCNClassifier
from models.transformer import TransformerClassifier
from models.mlp import MLPClassifier

def train_epoch_gcn(model, dataloader, criterion, optimizer, device):
    """Training epoch for GCN"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Training GCN")):
        # Handle both old format (graph, labels) and new format (graph, labels, metadata)
        if len(batch_data) == 3:
            graph, labels, metadata = batch_data
        else:
            graph, labels = batch_data
            
        if graph is None:
            continue
            
        graph = graph.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(graph)
        loss = criterion(logits, labels)
        
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print step loss every 100 steps
        if batch_idx % 100 == 0:
            print(f"  Step {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
        
        pred = torch.argmax(logits, dim=1)
        predictions.extend(pred.cpu().numpy())
        targets.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(targets, predictions)
    
    print(f"  Training Loss: {avg_loss:.6f}, Training Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy

def train_epoch_transformer(model, dataloader, criterion, optimizer, device):
    """Training epoch for Transformer"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Training Transformer")):
        # Handle both old format (token_ids, attention_mask, labels) and new format (token_ids, attention_mask, labels, metadata)
        if len(batch_data) == 4:
            token_ids, attention_mask, labels, metadata = batch_data
        else:
            token_ids, attention_mask, labels = batch_data
            
        token_ids = token_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(token_ids, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print step loss every 100 steps
        if batch_idx % 100 == 0:
            print(f"  Step {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
        
        pred = torch.argmax(logits, dim=1)
        predictions.extend(pred.cpu().numpy())
        targets.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(targets, predictions)
    
    print(f"  Training Loss: {avg_loss:.6f}, Training Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy

def train_epoch_mlp(model, dataloader, criterion, optimizer, device):
    """Training epoch for MLP"""
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    for batch_idx, (fingerprints, labels) in enumerate(tqdm(dataloader, desc="Training MLP")):
        fingerprints = fingerprints.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(fingerprints)
        loss = criterion(logits, labels)
        
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print step loss every 100 steps
        if batch_idx % 100 == 0:
            print(f"  Step {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
        
        pred = torch.argmax(logits, dim=1)
        predictions.extend(pred.cpu().numpy())
        targets.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(targets, predictions)
    
    print(f"  Training Loss: {avg_loss:.6f}, Training Accuracy: {accuracy:.4f}")
    
    return avg_loss, accuracy

def evaluate_gcn(model, dataloader, criterion, device, is_validation=False, original_data=None):
    """Evaluation for GCN with compound SMILES handling"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    probabilities = []
    metadata_list = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating GCN"):
            if len(batch_data) == 3:  # New format with metadata
                graph, labels, batch_metadata = batch_data
                if graph is None:
                    continue
                metadata_list.extend(batch_metadata)
            else:  # Old format
                graph, labels = batch_data
                if graph is None:
                    continue
                # Create dummy metadata for backward compatibility
                batch_size = labels.size(0)
                dummy_metadata = [{'original_idx': i, 'compound_variant': 0, 'total_variants': 1} 
                                for i in range(batch_size)]
                metadata_list.extend(dummy_metadata)
                
            graph = graph.to(device)
            labels = labels.to(device)
            
            logits = model(graph)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            pred = torch.argmax(logits, dim=1)
            prob = torch.softmax(logits, dim=1)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            probabilities.extend(prob[:, 1].cpu().numpy())  # Get positive class probability
    
    # Handle compound SMILES aggregation for validation
    if is_validation and original_data:
        agg_probs, orig_indices = aggregate_compound_predictions(probabilities, metadata_list, original_data)
        agg_targets, _ = aggregate_compound_targets(targets, metadata_list, original_data)
        agg_predictions = (agg_probs > 0.5).astype(int)
        
        return calculate_metrics_with_probs(total_loss, len(dataloader), agg_targets, agg_predictions, agg_probs)
    else:
        return calculate_metrics(total_loss, len(dataloader), targets, predictions, probabilities)

def evaluate_transformer(model, dataloader, criterion, device, is_validation=False, original_data=None):
    """Evaluation for Transformer with compound SMILES handling"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    probabilities = []
    metadata_list = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating Transformer"):
            if len(batch_data) == 4:  # New format with metadata
                token_ids, attention_mask, labels, batch_metadata = batch_data
                metadata_list.extend(batch_metadata)
            else:  # Old format
                token_ids, attention_mask, labels = batch_data
                # Create dummy metadata for backward compatibility
                batch_size = labels.size(0)
                dummy_metadata = [{'original_idx': i, 'compound_variant': 0, 'total_variants': 1} 
                                for i in range(batch_size)]
                metadata_list.extend(dummy_metadata)
            
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            logits = model(token_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            pred = torch.argmax(logits, dim=1)
            prob = torch.softmax(logits, dim=1)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            probabilities.extend(prob[:, 1].cpu().numpy())  # Get positive class probability
    
    # Handle compound SMILES aggregation for validation
    if is_validation and original_data:
        agg_probs, orig_indices = aggregate_compound_predictions(probabilities, metadata_list, original_data)
        agg_targets, _ = aggregate_compound_targets(targets, metadata_list, original_data)
        agg_predictions = (agg_probs > 0.5).astype(int)
        
        return calculate_metrics_with_probs(total_loss, len(dataloader), agg_targets, agg_predictions, agg_probs)
    else:
        return calculate_metrics(total_loss, len(dataloader), targets, predictions, probabilities)

def calculate_metrics(total_loss, num_batches, targets, predictions, probabilities):
    """Calculate evaluation metrics"""
    avg_loss = total_loss / num_batches
    
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')
    
    # ROC AUC (for binary classification)
    if len(set(targets)) == 2:
        probs = np.array(probabilities)
        auc_score = roc_auc_score(targets, probs[:, 1])
    else:
        auc_score = None
    
    return avg_loss, accuracy, precision, recall, f1, auc_score

def calculate_metrics_with_probs(total_loss, num_batches, targets, predictions, probabilities):
    """Calculate evaluation metrics when probabilities are already computed"""
    avg_loss = total_loss / num_batches
    
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')
    
    # ROC AUC (for binary classification)
    if len(set(targets)) == 2:
        auc_score = roc_auc_score(targets, probabilities)
    else:
        auc_score = None
    
    return avg_loss, accuracy, precision, recall, f1, auc_score

def aggregate_compound_predictions(probabilities, metadata_list, original_data):
    """Aggregate predictions for compound SMILES by taking the maximum score"""
    # Group predictions by original index
    grouped_preds = {}
    for prob, metadata in zip(probabilities, metadata_list):
        orig_idx = metadata['original_idx']
        variant = metadata['compound_variant']
        
        if orig_idx not in grouped_preds:
            grouped_preds[orig_idx] = []
        
        grouped_preds[orig_idx].append((variant, prob))
    
    # Take maximum probability for each original sample
    aggregated_probs = []
    original_indices = []
    
    for orig_idx in sorted(grouped_preds.keys()):
        variants_probs = grouped_preds[orig_idx]
        # Take the variant with highest probability (for positive class)
        best_variant, best_prob = max(variants_probs, key=lambda x: x[1])
        aggregated_probs.append(best_prob)
        # Use orig_idx directly as it should correspond to the position in original_data
        original_indices.append(orig_idx)
    
    return np.array(aggregated_probs), original_indices

def aggregate_compound_targets(targets, metadata_list, original_data):
    """Aggregate targets/labels for compound SMILES by taking the original label"""
    # Group targets by original index
    grouped_targets = {}
    for target, metadata in zip(targets, metadata_list):
        orig_idx = metadata['original_idx']
        
        if orig_idx not in grouped_targets:
            grouped_targets[orig_idx] = target
        # For targets, we expect all variants of the same compound to have the same label
        # So we can just take the first one or verify they're all the same
        assert grouped_targets[orig_idx] == target, f"Inconsistent labels for compound {orig_idx}: {grouped_targets[orig_idx]} vs {target}"
    
    # Get aggregated targets in the same order as original indices
    aggregated_targets = []
    original_indices = []
    
    for orig_idx in sorted(grouped_targets.keys()):
        aggregated_targets.append(grouped_targets[orig_idx])
        original_indices.append(orig_idx)
    
    return np.array(aggregated_targets), original_indices

def calculate_cluster_metrics(df_val, probabilities, top_n):
    """Calculate cluster-based metrics for top N compounds"""
    # pdb.set_trace()  # Add debug breakpoint here
    all_clusters = df_val[df_val["Label"] == 1].drop_duplicates("CLUSTER_LABEL").shape[0]
    
    sorted_indices = probabilities.argsort()[::-1]
    selection = df_val.iloc[sorted_indices[:top_n]].copy()
    selection["Score"] = probabilities[sorted_indices[:top_n]]
    
    hits = selection[selection["Label"] == 1]
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

def get_fingerprint_dim(fp_type):
    """Get fingerprint dimension based on type"""
    if fp_type == 'MACCS':
        return 167
    elif fp_type == 'RDK':
        return 2048
    elif fp_type == 'AVALON':
        return 2048
    elif fp_type == 'ATOMPAIR':
        return 2048
    else:
        raise ValueError(f"Unsupported fingerprint type: {fp_type}")

def update_results_csv(parent_dir, model_type, data_type, prauc, roc_auc, metric,
                       hits_50=None, clusters_50=None, cluster_prauc_50=None,
                       hits_200=None, clusters_200=None, cluster_prauc_200=None,
                       hits_500=None, clusters_500=None, cluster_prauc_500=None):
    """Update the results CSV file with enhanced metrics"""
    import datetime
    
    results_file = os.path.join(parent_dir, "model_results.csv")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = {
        "Timestamp": timestamp,
        "ModelType": model_type,
        "DataType": data_type,  # Changed from FingerprintType to DataType
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

def main():
    parser = argparse.ArgumentParser(description='GCN/Transformer/MLP Binder Classifier')
    parser.add_argument('--train_csv', type=str, required=True, help='Training CSV file')
    parser.add_argument('--val_csv', type=str, required=True, help='Validation CSV file')
    parser.add_argument('--smiles_col', type=str, default='SMILES', help='SMILES column name')
    parser.add_argument('--label_col', type=str, default='Label', help='Label column name')
    parser.add_argument('--model_type', type=str, default='gcn', choices=['gcn', 'transformer', 'mlp'], 
                        help='Model type: gcn, transformer, or mlp')
    
    # Add skip training flag
    parser.add_argument('--skip_training', action='store_true', 
                        help='Skip training and only run validation with randomly initialized model')
    
    # Add patience parameter for early stopping
    parser.add_argument('--patience', type=int, default=10, 
                        help='Number of epochs to wait before early stopping if no improvement')
    
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
        
        train_epoch_fn = train_epoch_gcn
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
        
        train_epoch_fn = train_epoch_transformer
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
        
        train_epoch_fn = train_epoch_mlp
        evaluate_fn = None  # MLP uses different evaluation logic
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if not args.skip_training:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
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
            
            print(f"Top 50 - Hits: {n_hits_50}, Clusters: {clusters_50}, Cluster PRAUC: {cluster_prauc_50:.4f}")
            print(f"Top 200 - Hits: {n_hits_200}, Clusters: {clusters_200}, Cluster PRAUC: {cluster_prauc_200:.4f}")
            print(f"Top 500 - Hits: {n_hits_500}, Clusters: {clusters_500}, Cluster PRAUC: {cluster_prauc_500:.4f}")
            
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
            
            print(f"Top 50 - Hits: {n_hits_50}, Clusters: {clusters_50}, Cluster PRAUC: {cluster_prauc_50:.4f}")
            print(f"Top 200 - Hits: {n_hits_200}, Clusters: {clusters_200}, Cluster PRAUC: {cluster_prauc_200:.4f}")
            print(f"Top 500 - Hits: {n_hits_500}, Clusters: {clusters_500}, Cluster PRAUC: {cluster_prauc_500:.4f}")
        
        print("Validation completed!")
        return
    
    # Training loop (only if not skipping training)
    best_val_prauc = 0
    best_val_acc = 0
    best_clusters_50 = 0  # Change from best_cluster_prauc_50 to best_clusters_50
    patience_counter = 0  # Add patience counter
    
    print("Starting training...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Training
        train_loss, train_acc = train_epoch_fn(model, train_loader, criterion, optimizer, device)
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
            
            print(f"Top 50 - Hits: {n_hits_50}, Clusters: {clusters_50}, Cluster PRAUC: {cluster_prauc_50:.4f}")
            print(f"Top 200 - Hits: {n_hits_200}, Clusters: {clusters_200}, Cluster PRAUC: {cluster_prauc_200:.4f}")
            print(f"Top 500 - Hits: {n_hits_500}, Clusters: {clusters_500}, Cluster PRAUC: {cluster_prauc_500:.4f}")
            
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
            
            print(f"Top 50 - Hits: {n_hits_50}, Clusters: {clusters_50}, Cluster PRAUC: {cluster_prauc_50:.4f}")
            print(f"Top 200 - Hits: {n_hits_200}, Clusters: {clusters_200}, Cluster PRAUC: {cluster_prauc_200:.4f}")
            print(f"Top 500 - Hits: {n_hits_500}, Clusters: {clusters_500}, Cluster PRAUC: {cluster_prauc_500:.4f}")
        
        # Save best model and check for early stopping - use clusters_50 as metric
        improved = False
        if clusters_50 > best_clusters_50:
            best_clusters_50 = clusters_50
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'best_{args.model_type}_model.pth'))
            print(f"Best model saved! New best clusters_50: {clusters_50}")
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
            
            predictions_path = os.path.join(args.save_dir, f"best_{args.model_type}_predictions.csv")
            df_predictions_val.to_csv(predictions_path, index=False)
            print(f"Best predictions saved to {predictions_path}")
        
        # Early stopping logic
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (current clusters_50: {clusters_50}, best: {best_clusters_50})")
        
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement")
            break
        
        # Update learning rate scheduler if using one
        if 'scheduler' in locals():
            scheduler.step(clusters_50)  # Use clusters_50 for scheduler too
    
    print(f"\nTraining completed. Best clusters_50: {best_clusters_50}")
    
    # Update results CSV only at the end of training with the best metrics
    parent_dir = os.path.dirname(os.path.abspath(args.save_dir))
    update_results_csv(
        parent_dir, f"{args.model_type.upper()}", 
        data_type,
        best_val_prauc, best_val_roc_auc, "clusters_50",
        best_n_hits_50, best_clusters_50_for_csv, best_cluster_prauc_50,
        best_n_hits_200, best_clusters_200, best_cluster_prauc_200,
        best_n_hits_500, best_clusters_500, best_cluster_prauc_500
    )
    
    # Final summary of best results
    print(f"\n{'='*60}")
    print(f"FINAL BEST RESULTS FOR {args.model_type.upper()}")
    print(f"{'='*60}")
    print(f"Best clusters_50: {best_clusters_50}")
    if 'best_val_prauc' in locals():
        print(f"Best PRAUC: {best_val_prauc:.4f}")
        print(f"Best ROC-AUC: {best_val_roc_auc:.4f}")
    print(f"Model saved to: {os.path.join(args.save_dir, f'best_{args.model_type}_model.pth')}")
    print(f"Predictions saved to: {os.path.join(args.save_dir, f'best_{args.model_type}_predictions.csv')}")

if __name__ == "__main__":
    main()