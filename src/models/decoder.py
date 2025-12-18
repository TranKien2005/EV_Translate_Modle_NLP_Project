"""
Modern Transformer Decoder.
Uses RoPE, RMSNorm, and SwiGLU for better performance.
"""

import torch
import torch.nn as nn
from typing import Optional

from .layers import DecoderLayer
from .rmsnorm import RMSNorm
from .rope import RotaryPositionalEmbedding


class TransformerDecoder(nn.Module):
    """
    Modern Transformer Decoder with improvements:
    - RMSNorm instead of LayerNorm
    - SwiGLU in FFN
    - RoPE for positional encoding
    - Pre-normalization for stability
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
        padding_idx: int = 0,
        use_rope: bool = True
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
            use_rope: Whether to use Rotary PE (if False, uses learned PE)
        """
        super().__init__()
        
        self.d_model = d_model
        self.use_rope = use_rope
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        
        # Positional encoding
        if use_rope:
            # RoPE is applied in attention
            self.rope = RotaryPositionalEmbedding(d_model // num_heads, max_seq_len)
        else:
            # Fallback: learned positional embedding
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, use_swiglu=True)
            for _ in range(num_layers)
        ])
        
        # Final RMSNorm (required for Pre-LN architecture)
        self.final_norm = RMSNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model ** -0.5)
        # Zero out padding embedding
        if self.embedding.padding_idx is not None:
            self.embedding.weight.data[self.embedding.padding_idx].zero_()
    
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
        seq_len = tgt.size(1)
        
        # Embedding + scale
        x = self.embedding(tgt) * (self.d_model ** 0.5)
        
        # Add positional encoding (if not using RoPE)
        if not self.use_rope:
            positions = torch.arange(seq_len, device=tgt.device).unsqueeze(0)
            x = x + self.pos_embedding(positions)
        
        x = self.dropout(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x
