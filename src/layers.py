"""
Transformer Layers: Encoder and Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    - Multi-Head Self-Attention
    - Add & Norm
    - Feed-Forward Network
    - Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: Optional mask [batch_size, 1, seq_len, seq_len]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    - Masked Multi-Head Self-Attention
    - Add & Norm
    - Multi-Head Cross-Attention (Encoder-Decoder Attention)
    - Add & Norm
    - Feed-Forward Network
    - Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Masked multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Multi-head cross-attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input [batch_size, tgt_seq_len, d_model]
            encoder_output: Encoder output [batch_size, src_seq_len, d_model]
            src_mask: Source mask [batch_size, 1, 1, src_seq_len]
            tgt_mask: Target mask (look-ahead) [batch_size, 1, tgt_seq_len, tgt_seq_len]
        Returns:
            output: [batch_size, tgt_seq_len, d_model]
        """
        # Masked self-attention with residual connection and layer norm
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = x + self.dropout1(self_attn_output)
        x = self.norm1(x)
        
        # Cross-attention with residual connection and layer norm
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout2(cross_attn_output)
        x = self.norm2(x)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)
        
        return x
