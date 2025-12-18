"""
Full Transformer Model for Sequence-to-Sequence Translation.
Combines Encoder and Decoder with output projection.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    
    Architecture:
        - Source Embedding + Positional Encoding
        - Encoder (stack of N layers)
        - Target Embedding + Positional Encoding
        - Decoder (stack of N layers)
        - Output Linear Projection
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        src_padding_idx: int = 0,
        tgt_padding_idx: int = 0,
        share_embeddings: bool = False,
        weight_tying: bool = True  # New: tie decoder embedding with output projection
    ):
        """
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension (embedding size)
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            src_padding_idx: Source padding token index
            tgt_padding_idx: Target padding token index
            share_embeddings: Whether to share embeddings between encoder and decoder
            weight_tying: Whether to tie decoder embedding with output projection
        """
        super().__init__()
        
        self.src_padding_idx = src_padding_idx
        self.tgt_padding_idx = tgt_padding_idx
        self.d_model = d_model
        self.weight_tying = weight_tying
        
        # Encoder
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            padding_idx=src_padding_idx
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            padding_idx=tgt_padding_idx
        )
        
        # Output projection (no bias for weight tying)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size, bias=not weight_tying)
        
        # Share embeddings if specified
        if share_embeddings:
            self.decoder.embedding.weight = self.encoder.embedding.weight
        
        # Weight Tying: share decoder embedding with output projection
        # This reduces parameters and improves training
        # Paper: "Using the Output Embedding to Improve Language Models" (Press & Wolf, 2017)
        if weight_tying:
            self.output_projection.weight = self.decoder.embedding.weight
        
        # Initialize output projection (only if not using weight tying)
        if not weight_tying:
            self._init_weights()
    
    def _init_weights(self):
        """Initialize output projection weights."""
        nn.init.xavier_uniform_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)
    
    def create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create source padding mask.
        
        Args:
            src: Source tensor of shape (batch, src_seq_len)
        
        Returns:
            Mask of shape (batch, 1, 1, src_seq_len)
        """
        # (batch, 1, 1, src_seq_len)
        src_mask = (src != self.src_padding_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def create_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Create target mask (padding mask + causal mask).
        
        Args:
            tgt: Target tensor of shape (batch, tgt_seq_len)
        
        Returns:
            Combined mask of shape (batch, 1, tgt_seq_len, tgt_seq_len)
        """
        batch_size, tgt_len = tgt.size()
        
        # Padding mask: (batch, 1, 1, tgt_seq_len)
        padding_mask = (tgt != self.tgt_padding_idx).unsqueeze(1).unsqueeze(2)
        
        # Causal mask (lower triangular): (1, 1, tgt_seq_len, tgt_seq_len)
        causal_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=tgt.device)
        ).unsqueeze(0).unsqueeze(0)
        
        # Combine masks
        tgt_mask = padding_mask & causal_mask.bool()
        
        return tgt_mask
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer.
        
        Args:
            src: Source token IDs of shape (batch, src_seq_len)
            tgt: Target token IDs of shape (batch, tgt_seq_len)
            src_mask: Optional source mask
            tgt_mask: Optional target mask (will be created if None)
        
        Returns:
            Output logits of shape (batch, tgt_seq_len, tgt_vocab_size)
        """
        # Create masks if not provided
        if src_mask is None:
            src_mask = self.create_src_mask(src)
        if tgt_mask is None:
            tgt_mask = self.create_tgt_mask(tgt)
        
        # Encode
        encoder_output = self.encoder(src, src_mask)
        
        # Decode
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        return output
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source sequence."""
        if src_mask is None:
            src_mask = self.create_src_mask(src)
        return self.encoder(src, src_mask)
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode with encoder output."""
        if tgt_mask is None:
            tgt_mask = self.create_tgt_mask(tgt)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_projection(decoder_output)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        return (
            f"Transformer(\n"
            f"  encoder_layers={len(self.encoder.layers)},\n"
            f"  decoder_layers={len(self.decoder.layers)},\n"
            f"  d_model={self.d_model},\n"
            f"  parameters={self.count_parameters():,}\n"
            f")"
        )
