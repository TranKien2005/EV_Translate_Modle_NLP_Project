"""
RMSNorm (Root Mean Square Layer Normalization).
Simpler and faster than LayerNorm - used in LLaMA, Qwen, Mistral.

Paper: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Unlike LayerNorm, RMSNorm:
    - Does not re-center (no mean subtraction)
    - Does not have bias parameter
    - Faster computation (10-15% speedup)
    
    Formula: x_norm = x / RMS(x) * gamma
    where RMS(x) = sqrt(mean(x^2) + eps)
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Normalized tensor of same shape
        """
        # RMS = sqrt(mean(x^2))
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
