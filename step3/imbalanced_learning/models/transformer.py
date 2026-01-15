import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, dropout_rate=0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(attention_probs_dropout)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Take the dot product between "query" and "key" to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # Apply the attention mask
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, 
                 attention_probs_dropout=0.1, hidden_dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_attention_heads, attention_probs_dropout)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(hidden_dropout_rate)
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = F.relu
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout_rate)
        self.output_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_layer_norm(attention_output + hidden_states)
        
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.intermediate_act_fn(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        layer_output = self.output_layer_norm(layer_output + attention_output)
        
        return layer_output

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, intermediate_size, num_attention_heads,
                 attention_probs_dropout=0.1, hidden_dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, intermediate_size, num_attention_heads,
                           attention_probs_dropout, hidden_dropout_rate)
            for _ in range(num_layers)
        ])
        
    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size=65, max_position_embeddings=1000, 
                 hidden_size=64, num_layers=8, intermediate_size=512,
                 num_attention_heads=8, attention_probs_dropout=0.1,
                 hidden_dropout_rate=0.1, num_classes=2):
        super().__init__()
        
        self.embeddings = Embeddings(vocab_size, hidden_size, max_position_embeddings, hidden_dropout_rate)
        self.encoder = TransformerEncoder(num_layers, hidden_size, intermediate_size, 
                                        num_attention_heads, attention_probs_dropout, hidden_dropout_rate)
        
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None:
            # Convert attention mask to the format expected by the model
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
            
        embedding_output = self.embeddings(input_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask)
        
        # Use the [CLS] token representation (first token)
        pooled_output = self.pooler(encoder_outputs[:, 0])
        logits = self.classifier(pooled_output)
        
        return logits

    def get_features(self, token_ids, attention_mask):
        """Extract features before the final classification layer"""
        # Ensure inputs are on the same device as the model
        device = next(self.parameters()).device
        token_ids = token_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        if attention_mask is not None:
            # Convert attention mask to the format expected by the model
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
            
        # Get embeddings using the correct attribute name
        embedding_output = self.embeddings(token_ids)
        
        # Pass through encoder (transformer layers)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask)
        
        # Use the [CLS] token representation (first token) and apply pooler
        pooled_output = self.pooler(encoder_outputs[:, 0])
        
        return pooled_output