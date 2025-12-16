"""
Transformer Encoder and Decoder Layers.
Includes Feed-Forward Network and Layer components.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (usually 4 * d_model)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer with Pre-LayerNorm.
    
    Pre-LN Structure (more stable training):
        1. LayerNorm -> Multi-Head Self-Attention -> Residual
        2. LayerNorm -> Feed-Forward -> Residual
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            src_mask: Source attention mask
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pre-LayerNorm: normalize before sublayer
        # Self-attention
        normalized = self.norm1(x)
        attn_output, _ = self.self_attention(normalized, normalized, normalized, src_mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward
        normalized = self.norm2(x)
        ff_output = self.feed_forward(normalized)
        x = x + self.dropout(ff_output)
        
        return x


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer with Pre-LayerNorm.
    
    Pre-LN Structure (more stable training):
        1. LayerNorm -> Masked Multi-Head Self-Attention -> Residual
        2. LayerNorm -> Multi-Head Cross-Attention -> Residual
        3. LayerNorm -> Feed-Forward -> Residual
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Masked self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention (encoder-decoder attention)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input of shape (batch, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch, src_seq_len, d_model)
            src_mask: Source attention mask (for cross-attention)
            tgt_mask: Target attention mask (for self-attention, causal mask)
        
        Returns:
            Output tensor of shape (batch, tgt_seq_len, d_model)
        """
        # Pre-LayerNorm: normalize before each sublayer
        
        # Masked self-attention
        normalized = self.norm1(x)
        self_attn_output, _ = self.self_attention(normalized, normalized, normalized, tgt_mask)
        x = x + self.dropout(self_attn_output)
        
        # Cross-attention (query=decoder, key/value=encoder)
        normalized = self.norm2(x)
        cross_attn_output, _ = self.cross_attention(normalized, encoder_output, encoder_output, src_mask)
        x = x + self.dropout(cross_attn_output)
        
        # Feed-forward
        normalized = self.norm3(x)
        ff_output = self.feed_forward(normalized)
        x = x + self.dropout(ff_output)
        
        return x
