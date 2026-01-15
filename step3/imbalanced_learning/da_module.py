import torch
import numpy as np
from sklearn.metrics import accuracy_score

class DataAugmentationCombiner:
    def __init__(self, device, num_class_list=None, alpha=1.0, remix_kappa=3.0, remix_tau=0.5):
        """
        Data augmentation combiner for Mixup and Remix
        
        Args:
            device: PyTorch device
            num_class_list: List of class sample counts [class_0_count, class_1_count]
            alpha: Beta distribution parameter for mixing coefficient
            remix_kappa: Remix kappa parameter for class imbalance sensitivity
            remix_tau: Remix tau parameter for threshold
        """
        self.device = device
        self.num_class_list = num_class_list
        self.alpha = alpha
        self.remix_kappa = remix_kappa
        self.remix_tau = remix_tau
    
    def mixup(self, model, criterion, data, labels, model_type):
        """
        Mixup implementation for different model types
        
        Args:
            model: The model with feature extraction and classification head
            criterion: Loss function
            data: Input data (varies by model type)
            labels: Target labels
            model_type: 'gcn', 'transformer', or 'mlp'
        
        Returns:
            loss: Mixed loss
            accuracy: Mixed accuracy
        """
        # Sample mixing coefficient
        l = np.random.beta(self.alpha, self.alpha)
        
        # Extract features based on model type
        feat = self._get_features(model, data, model_type)
        
        # Random permutation for mixing
        idx = torch.randperm(feat.size(0))
        feat_a = feat
        feat_b = feat[idx]
        
        # Mix features
        mixed_feat = l * feat_a + (1 - l) * feat_b
        
        # Mix labels
        labels = labels.to(self.device)
        label_a = labels
        label_b = labels[idx]
        
        # Forward through classification head
        if hasattr(model, 'classify_from_features'):
            output = model.classify_from_features(mixed_feat)
        else:
            output = self._classify_from_features(model, mixed_feat, model_type)
        
        # Compute mixed loss
        loss = l * criterion(output, label_a) + (1 - l) * criterion(output, label_b)
        
        # Calculate mixed accuracy
        predictions = torch.argmax(output, dim=1)
        acc_a = accuracy_score(label_a.cpu().numpy(), predictions.cpu().numpy())
        acc_b = accuracy_score(label_b.cpu().numpy(), predictions.cpu().numpy())
        mixed_acc = l * acc_a + (1 - l) * acc_b
        
        return loss, mixed_acc
    
    def remix(self, model, criterion, data, labels, model_type):
        """
        Remix implementation for different model types
        
        Args:
            model: The model with feature extraction and classification head
            criterion: Loss function
            data: Input data (varies by model type)
            labels: Target labels
            model_type: 'gcn', 'transformer', or 'mlp'
        
        Returns:
            loss: Adaptive mixed loss
            accuracy: Mixed accuracy
        """
        assert self.num_class_list is not None, "num_class_list is required for Remix"
        
        # Sample base mixing coefficient
        l = np.random.beta(self.alpha, self.alpha)
        
        # Extract features (same as mixup)
        feat = self._get_features(model, data, model_type)
        
        # Random permutation
        idx = torch.randperm(feat.size(0))
        feat_a = feat
        feat_b = feat[idx]
        
        # Mix features
        mixed_feat = l * feat_a + (1 - l) * feat_b
        
        # Mix labels
        labels = labels.to(self.device)
        label_a = labels
        label_b = labels[idx]
        
        # Adaptive lambda based on class imbalance (key difference from Mixup)
        l_list = torch.empty(feat.shape[0]).fill_(l).float().to(self.device)
        
        # Get class frequencies
        n_i = torch.tensor([self.num_class_list[int(label)] for label in label_a], device=self.device)
        n_j = torch.tensor([self.num_class_list[int(label)] for label in label_b], device=self.device)
        
        # Adjust mixing coefficients based on class frequencies
        if l < self.remix_tau:
            l_list[n_i / n_j >= self.remix_kappa] = 0
        if 1 - l < self.remix_tau:
            l_list[(n_i * self.remix_kappa) / n_j <= 1] = 1
        
        # Forward through classification head
        if hasattr(model, 'classify_from_features'):
            output = model.classify_from_features(mixed_feat)
        else:
            output = self._classify_from_features(model, mixed_feat, model_type)
        
        # Compute adaptive mixed loss
        loss_a = criterion(output, label_a)
        loss_b = criterion(output, label_b)
        
        # Handle different loss shapes
        if loss_a.dim() == 0:  # Scalar loss
            loss = l_list.mean() * loss_a + (1 - l_list.mean()) * loss_b
        else:  # Per-sample loss
            loss = (l_list * loss_a + (1 - l_list) * loss_b).mean()
        
        # Calculate mixed accuracy
        predictions = torch.argmax(output, dim=1)
        acc_a = accuracy_score(label_a.cpu().numpy(), predictions.cpu().numpy())
        acc_b = accuracy_score(label_b.cpu().numpy(), predictions.cpu().numpy())
        mixed_acc = l_list.mean().item() * acc_a + (1 - l_list.mean().item()) * acc_b
        
        return loss, mixed_acc
    
    def _get_features(self, model, inputs, model_type):
        """Extract features from model before final classification"""
        # Get model device
        device = next(model.parameters()).device
        
        if model_type == 'gcn':
            # For GCN, inputs is a DGL graph
            inputs = inputs.to(device)
            if hasattr(model, 'get_features'):
                return model.get_features(inputs)
            else:
                # Fallback: run through model layers except classifier
                h = inputs.ndata['feat']
                
                # Pass through all layers except the last one
                for i, layer in enumerate(model.layers[:-1]):
                    if i > 0 and hasattr(model, 'dropout'):
                        h = model.dropout(h)
                    h = layer(inputs, h)
                    h = torch.relu(h)
                
                # Apply final dropout but not classification
                if hasattr(model, 'dropout'):
                    h = model.dropout(h)
                    
                # Apply graph-level pooling
                import dgl
                graph_feat = dgl.mean_nodes(inputs, h)
                return graph_feat
                
        elif model_type == 'transformer':
            token_ids, attention_mask = inputs
            # Move inputs to device
            token_ids = token_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            if hasattr(model, 'get_features'):
                return model.get_features(token_ids, attention_mask)
            else:
                # Fallback: extract features before classifier
                embeddings = model.embeddings(token_ids)
                
                if attention_mask is not None:
                    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    extended_attention_mask = extended_attention_mask.to(dtype=next(model.parameters()).dtype)
                    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
                else:
                    extended_attention_mask = None
                
                encoder_outputs = model.encoder(embeddings, extended_attention_mask)
                pooled_output = model.pooler(encoder_outputs[:, 0])
                
                return pooled_output
                
        elif model_type == 'mlp':
            # For MLP, inputs is a tensor
            inputs = inputs.to(device)
            if hasattr(model, 'get_features'):
                return model.get_features(inputs)
            else:
                # Fallback: run through model layers except classifier
                x = inputs
                layers = list(model.children())
                for layer in layers[:-1]:  # Exclude final classifier
                    x = layer(x)
                return x
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _extract_gcn_features(self, model, graph):
        """Extract features from GCN model"""
        # Temporarily modify model to return features
        original_forward = model.forward
        features = None
        
        def feature_hook(module, input, output):
            nonlocal features
            features = output
        
        # Find the last layer before classification
        layers = list(model.children())
        if len(layers) > 1:
            # Hook before the final layer
            handle = layers[-2].register_forward_hook(feature_hook)
            _ = model(graph)
            handle.remove()
        else:
            # Fallback: use full forward pass
            features = model(graph)
        
        return features
    
    def _extract_transformer_features(self, model, token_ids, attention_mask):
        """Extract features from Transformer model"""
        # Similar approach for transformer
        features = None
        
        def feature_hook(module, input, output):
            nonlocal features
            features = output
        
        # Find the layer before final classification
        if hasattr(model, 'transformer') and hasattr(model, 'classifier'):
            handle = model.transformer.register_forward_hook(feature_hook)
            _ = model(token_ids, attention_mask)
            handle.remove()
        else:
            # Fallback
            features = model(token_ids, attention_mask)
        
        return features
    
    def _extract_mlp_features(self, model, data):
        """Extract features from MLP model"""
        features = None
        
        def feature_hook(module, input, output):
            nonlocal features
            features = output
        
        # Hook before the final layer
        layers = list(model.children())
        if len(layers) > 1:
            handle = layers[-2].register_forward_hook(feature_hook)
            _ = model(data)
            handle.remove()
        else:
            features = model(data)
        
        return features
    
    def _classify_from_features(self, model, features, model_type):
        """Apply classification head to features"""
        if model_type == 'gcn':
            # For GCN, apply the final layer (classifier)
            if hasattr(model, 'classifier'):
                return model.classifier(features)
            else:
                # Fallback: use the last layer
                return model.layers[-1](features)
        elif model_type == 'transformer':
            # For Transformer, apply the classifier
            if hasattr(model, 'classifier'):
                return model.classifier(features)
            else:
                # Fallback: look for final linear layer
                for layer in reversed(list(model.modules())):
                    if isinstance(layer, torch.nn.Linear):
                        return layer(features)
        elif model_type == 'mlp':
            # For MLP, apply the final layer
            if hasattr(model, 'classifier'):
                return model.classifier(features)
            else:
                # Fallback: use the last layer
                layers = list(model.children())
                if len(layers) > 0:
                    return layers[-1](features)
        
        # Fallback: return features as-is (assume they're logits)
        return features