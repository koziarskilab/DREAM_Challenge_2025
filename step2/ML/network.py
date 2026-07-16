import torch
import torch.nn as nn


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
    
    def forward_features(self, x):
        """Return features from the second-to-last layer"""
        return self.model[:-1](x)


class BBNModel(nn.Module):
    """Bilateral-Branch Network for imbalanced learning"""
    def __init__(self, input_dim, hidden_dims=[256, 256, 256], dropout=0.3, num_classes=2):
        super(BBNModel, self).__init__()
        
        # Shared feature extractor
        feature_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims[:-1]:  # All but last hidden layer
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            feature_layers.append(nn.ReLU())
            feature_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # Conventional branch (for balanced data)
        self.conventional_branch = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], num_classes)
        )
        
        # Re-balancing branch (for imbalanced data)
        self.rebalancing_branch = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x, branch="both"):
        features = self.feature_extractor(x)
        
        if branch == "conventional":
            return self.conventional_branch(features)
        elif branch == "rebalancing":
            return self.rebalancing_branch(features)
        else:  # both branches
            conv_out = self.conventional_branch(features)
            rebal_out = self.rebalancing_branch(features)
            return conv_out, rebal_out
    
    def forward_features(self, x):
        """Return features from the feature extractor"""
        return self.feature_extractor(x)