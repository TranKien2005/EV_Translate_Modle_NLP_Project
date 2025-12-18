"""
Transformer Encoder and Decoder Layers.
Includes SwiGLU Feed-Forward Network and RMSNorm.

Modern improvements over original Transformer:
- SwiGLU activation (from PaLM, LLaMA)
- RMSNorm instead of LayerNorm (faster, equally effective)
- Pre-normalization for training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .attention import MultiHeadAttention
from .rmsnorm import RMSNorm


class SwiGLU(nn.Module):
    """
    SwiGLU activation function for Feed-Forward Network.
    
    Used in PaLM, LLaMA, Mistral - better than ReLU/GELU.
    
    SwiGLU(x) = (Swish(xW1) ⊙ xV) W2
    where Swish(x) = x * sigmoid(x)
    
    Paper: "GLU Variants Improve Transformer" (Shazeer, 2020)
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension (note: actual hidden = d_ff * 2/3 for same param count)
            dropout: Dropout rate
        """
        super().__init__()
        
        # For SwiGLU, we use 2/3 of d_ff for each of the two projections
        # This keeps parameter count similar to standard FFN
        hidden_dim = int(2 * d_ff / 3)
        # Make it multiple of 8 for efficiency
        hidden_dim = ((hidden_dim + 7) // 8) * 8
        
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)  # Down projection
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)  # Up projection
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # SwiGLU: (Swish(xW1) ⊙ xW3) W2
        # Swish(x) = x * sigmoid(x) = F.silu(x)
        gate = F.silu(self.w1(x))  # Swish activation
        up = self.w3(x)
        x = gate * up  # Element-wise multiplication
        x = self.dropout(x)
        x = self.w2(x)
        return x


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with SwiGLU.
    
    Modern FFN using SwiGLU activation instead of ReLU/GELU.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, use_swiglu: bool = True):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            use_swiglu: Whether to use SwiGLU (True) or GELU (False)
        """
        super().__init__()
        
        if use_swiglu:
            self.ffn = SwiGLU(d_model, d_ff, dropout)
        else:
            # Standard FFN with GELU
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        return self.ffn(x)


class EncoderLayer(nn.Module):
    """
    Modern Transformer Encoder Layer with Pre-RMSNorm.
    
    Improvements over original:
    - RMSNorm instead of LayerNorm (faster)
    - SwiGLU in FFN (better performance)
    - Pre-normalization (more stable training)
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1,
        use_swiglu: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            use_swiglu: Whether to use SwiGLU activation
        """
        super().__init__()
        
        # Self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = RMSNorm(d_model)
        
        # Feed-forward with SwiGLU
        self.feed_forward = FeedForward(d_model, d_ff, dropout, use_swiglu)
        self.norm2 = RMSNorm(d_model)
        
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
        # Pre-RMSNorm: normalize before sublayer
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
    Modern Transformer Decoder Layer with Pre-RMSNorm.
    
    Improvements over original:
    - RMSNorm instead of LayerNorm (faster)
    - SwiGLU in FFN (better performance)
    - Pre-normalization (more stable training)
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1,
        use_swiglu: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            use_swiglu: Whether to use SwiGLU activation
        """
        super().__init__()
        
        # Masked self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = RMSNorm(d_model)
        
        # Cross-attention (encoder-decoder attention)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        
        # Feed-forward with SwiGLU
        self.feed_forward = FeedForward(d_model, d_ff, dropout, use_swiglu)
        self.norm3 = RMSNorm(d_model)
        
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
        # Pre-RMSNorm: normalize before each sublayer
        
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
