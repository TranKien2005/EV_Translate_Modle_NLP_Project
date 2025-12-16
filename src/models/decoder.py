"""
Transformer Decoder.
Stack of Decoder Layers with embedding and positional encoding.
"""

import torch
import torch.nn as nn
from typing import Optional

from .layers import DecoderLayer
from .positional import PositionalEncoding


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder.
    
    Structure:
        1. Token Embedding
        2. Positional Encoding
        3. Stack of N Decoder Layers
        4. Output Linear Projection
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
            vocab_size: Target vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
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
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
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
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target token IDs of shape (batch, tgt_seq_len)
            encoder_output: Encoder output of shape (batch, src_seq_len, d_model)
            src_mask: Source attention mask (for cross-attention)
            tgt_mask: Target attention mask (causal mask)
        
        Returns:
            Decoder output of shape (batch, tgt_seq_len, d_model)
        """
        # Embedding + scale
        x = self.embedding(tgt) * (self.d_model ** 0.5)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
