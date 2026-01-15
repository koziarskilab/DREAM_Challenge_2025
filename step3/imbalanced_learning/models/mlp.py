import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union

class MLP(nn.Module):
    """MLP with linear output based on the codebase implementation"""
    
    def __init__(self, input_dim, output_dim, hidden_dims_lst, dropout=0.1):
        """
        input_dim (int): Input dimension
        output_dim (int): Output dimension  
        hidden_dims_lst (list): List of hidden layer dimensions
        dropout (float): Dropout rate
        """
        super(MLP, self).__init__()
        layer_size = len(hidden_dims_lst) + 1
        dims = [input_dim] + hidden_dims_lst + [output_dim]
        
        self.predictor = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)]
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, v):
        """Forward pass through MLP"""
        v = v.float()
        for i, layer in enumerate(self.predictor):
            if i == len(self.predictor) - 1:
                # Last layer - no activation
                v = layer(v)
            else:
                # Hidden layers with ReLU and dropout
                v = F.relu(layer(v))
                v = self.dropout(v)
        return v

class MLPClassifier(nn.Module):
    """MLP Classifier for molecular fingerprints"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64], n_classes=2, dropout=0.1):
        """
        Args:
            input_dim (int): Input fingerprint dimension
            hidden_dims (list): List of hidden layer dimensions
            n_classes (int): Number of output classes
            dropout (float): Dropout rate
        """
        super(MLPClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        
        # Build MLP layers
        self.classifier = MLP(
            input_dim=input_dim,
            output_dim=n_classes,
            hidden_dims_lst=hidden_dims,
            dropout=dropout
        )
        
    def forward(self, x):
        """Forward pass"""
        return self.classifier(x)