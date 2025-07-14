import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
import dgl.function as fn

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, gnn_norm="none", activation=None, 
                 residual=True, batchnorm=True, dropout=0.0):
        super(GCNLayer, self).__init__()
        
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.gnn_norm = gnn_norm
        self.activation = activation
        self.residual = residual
        self.batchnorm = batchnorm
        self.dropout = dropout
        
        # Core GCN layer
        self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats, 
                                  norm=gnn_norm, activation=activation)
        
        # Residual connection
        if residual and in_feats != out_feats:
            self.res_connection = nn.Linear(in_feats, out_feats)
        elif residual:
            self.res_connection = nn.Identity()
        
        # Batch normalization
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)
        
        # Dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, g, feats):
        # Graph convolution
        new_feats = self.graph_conv(g, feats)
        
        # Residual connection
        if self.residual:
            if hasattr(self, 'res_connection'):
                new_feats = new_feats + self.res_connection(feats)
            else:
                new_feats = new_feats + feats
        
        # Batch normalization
        if self.batchnorm:
            new_feats = self.bn_layer(new_feats)
        
        # Dropout
        if self.dropout > 0:
            new_feats = self.dropout_layer(new_feats)
        
        return new_feats

class WeightedSumAndMax(nn.Module):
    def __init__(self, in_feats):
        super(WeightedSumAndMax, self).__init__()
        self.in_feats = in_feats
        self.weight = nn.Parameter(torch.randn(in_feats, 1))
        self.bias = nn.Parameter(torch.randn(1))
    
    def forward(self, g, feats):
        with g.local_scope():
            g.ndata['h'] = feats
            # Weighted sum
            weight = torch.sigmoid(torch.matmul(feats, self.weight) + self.bias)
            g.ndata['w'] = weight
            weighted_sum = dgl.sum_nodes(g, 'h', 'w')
            
            # Max pooling
            max_pool = dgl.max_nodes(g, 'h')
            
            # Concatenate
            return torch.cat([weighted_sum, max_pool], dim=1)

class GCNClassifier(nn.Module):
    def __init__(self, in_feats, hidden_feats=[64, 64], n_classes=2, 
                 gnn_norm="none", activation=F.relu, residual=True, 
                 batchnorm=True, dropout=0.0):
        super(GCNClassifier, self).__init__()
        
        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.n_classes = n_classes
        self.n_layers = len(hidden_feats)
        
        # GCN layers
        self.gnn_layers = nn.ModuleList()
        
        # First layer
        self.gnn_layers.append(GCNLayer(
            in_feats=in_feats, 
            out_feats=hidden_feats[0],
            gnn_norm=gnn_norm,
            activation=activation,
            residual=residual,
            batchnorm=batchnorm,
            dropout=dropout
        ))
        
        # Hidden layers
        for i in range(1, self.n_layers):
            self.gnn_layers.append(GCNLayer(
                in_feats=hidden_feats[i-1],
                out_feats=hidden_feats[i],
                gnn_norm=gnn_norm,
                activation=activation,
                residual=residual,
                batchnorm=batchnorm,
                dropout=dropout
            ))
        
        # Graph-level readout
        gnn_out_feats = hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(gnn_out_feats * 2, gnn_out_feats),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn_out_feats, n_classes)
        )
    
    def forward(self, g):
        # Node features
        h = g.ndata['feat']
        
        # GCN layers
        for layer in self.gnn_layers:
            h = layer(g, h)
        
        # Graph-level readout
        graph_repr = self.readout(g, h)
        
        # Classification
        logits = self.classifier(graph_repr)
        
        return logits