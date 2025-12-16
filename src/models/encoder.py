"""
Transformer Encoder.
Stack of Encoder Layers with embedding and positional encoding.
"""

import torch
import torch.nn as nn
from typing import Optional

from .layers import EncoderLayer
from .positional import PositionalEncoding


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder.
    
    Structure:
        1. Token Embedding
        2. Positional Encoding
        3. Stack of N Encoder Layers
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        """
        Args:
            vocab_size: Source vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            padding_idx: Padding token index
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model ** -0.5)
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: Source token IDs of shape (batch, src_seq_len)
            src_mask: Source attention mask
        
        Returns:
            Encoder output of shape (batch, src_seq_len, d_model)
        """
        # Embedding + scale
        x = self.embedding(src) * (self.d_model ** 0.5)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x
