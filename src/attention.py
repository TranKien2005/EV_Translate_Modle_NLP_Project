"""
Attention Mechanisms for Transformer
Includes:
- Scaled Dot-Product Attention
- Multi-Head Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    """
    Scaled Dot-Product Attention
    
    Args:
        Q: Query tensor [batch_size, num_heads, seq_len_q, d_k]
        K: Key tensor [batch_size, num_heads, seq_len_k, d_k]
        V: Value tensor [batch_size, num_heads, seq_len_v, d_v]
        mask: Optional mask tensor [batch_size, 1, seq_len_q, seq_len_k]
        dropout: Optional dropout layer
    
    Returns:
        output: Attention output [batch_size, num_heads, seq_len_q, d_v]
        attention_weights: Attention weights [batch_size, num_heads, seq_len_q, seq_len_k]
    """
    d_k = Q.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Apply dropout
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k)
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        """
        Combine heads back to original shape
        Args:
            x: [batch_size, num_heads, seq_len, d_k]
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        """
        Forward pass
        Args:
            Q: Query [batch_size, seq_len_q, d_model]
            K: Key [batch_size, seq_len_k, d_model]
            V: Value [batch_size, seq_len_v, d_model]
            mask: Optional mask [batch_size, 1, seq_len_q, seq_len_k]
        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        # Linear projections
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Apply scaled dot-product attention
        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout)
        
        # Combine heads
        output = self.combine_heads(attention_output)
        
        # Final linear projection
        output = self.W_o(output)
        
        return output, attention_weights
