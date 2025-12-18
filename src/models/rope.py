"""
Rotary Positional Embedding (RoPE).
Used in LLaMA, Qwen, Mistral - better than sinusoidal PE.

Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    
    Key advantages over sinusoidal PE:
    - Encodes both absolute and relative positions
    - Better extrapolation to longer sequences
    - Decaying inter-token dependency with distance
    - Applied directly in attention (not added to embeddings)
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 2048, base: float = 10000.0):
        """
        Args:
            d_model: Model dimension (must be even)
            max_seq_len: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()
        
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        self._precompute_freqs(max_seq_len)
    
    def _precompute_freqs(self, seq_len: int):
        """Precompute frequency tensors for rotary embeddings."""
        # Compute inverse frequencies: 1 / (base^(2i/d))
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.d_model, 2).float() / self.d_model)
        )
        
        # Create position indices
        positions = torch.arange(seq_len).float()
        
        # Compute angles: position * inv_freq
        # Shape: (seq_len, d_model/2)
        angles = torch.outer(positions, inv_freq)
        
        # Compute cos and sin
        # Shape: (seq_len, d_model/2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        # Register as buffers (not trainable, but saved with model)
        self.register_buffer('cos_cached', cos, persistent=False)
        self.register_buffer('sin_cached', sin, persistent=False)
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply rotary positional embedding.
        
        Args:
            x: Input tensor of shape (batch, seq_len, num_heads, head_dim)
               or (batch, num_heads, seq_len, head_dim)
            offset: Position offset (for inference with KV cache)
        
        Returns:
            Tensor with rotary embeddings applied
        """
        seq_len = x.shape[-2] if x.dim() == 4 else x.shape[1]
        
        # Extend cache if needed
        if seq_len + offset > self.cos_cached.shape[0]:
            self._precompute_freqs(seq_len + offset)
            # Move to correct device
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)
        
        # Get relevant cos/sin values
        cos = self.cos_cached[offset:offset + seq_len]
        sin = self.sin_cached[offset:offset + seq_len]
        
        return self._apply_rotary(x, cos, sin)
    
    def _apply_rotary(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply rotary transformation.
        
        The rotation is applied as:
        x_rotated = x * cos + rotate_half(x) * sin
        
        where rotate_half swaps and negates pairs of elements.
        """
        # Split x into pairs for rotation
        # x shape: (..., head_dim) where head_dim = d_model // num_heads
        x1 = x[..., ::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices
        
        # Expand cos/sin to match x dimensions
        # cos/sin shape: (seq_len, d_model/2)
        while cos.dim() < x1.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        
        # Apply rotation
        # For each pair (x1, x2), rotate by angle theta:
        # x1' = x1 * cos - x2 * sin
        # x2' = x1 * sin + x2 * cos
        x1_rotated = x1 * cos - x2 * sin
        x2_rotated = x1 * sin + x2 * cos
        
        # Interleave back
        x_rotated = torch.stack([x1_rotated, x2_rotated], dim=-1).flatten(-2)
        
        return x_rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embedding to query and key tensors.
    
    This is a utility function for applying RoPE in attention.
    
    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
        cos: Cosine values of shape (seq_len, head_dim/2)
        sin: Sine values of shape (seq_len, head_dim/2)
    
    Returns:
        Tuple of rotated (query, key) tensors
    """
    # Expand dimensions for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation to query
    q1, q2 = q[..., ::2], q[..., 1::2]
    q_rotated = torch.stack([
        q1 * cos - q2 * sin,
        q1 * sin + q2 * cos
    ], dim=-1).flatten(-2)
    
    # Apply rotation to key
    k1, k2 = k[..., ::2], k[..., 1::2]
    k_rotated = torch.stack([
        k1 * cos - k2 * sin,
        k1 * sin + k2 * cos
    ], dim=-1).flatten(-2)
    
    return q_rotated, k_rotated
