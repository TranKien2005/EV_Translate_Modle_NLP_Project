"""
Helper utilities for training.
"""

import torch
import random
import numpy as np
from typing import Optional, Tuple


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✓ Set random seed: {seed}")


def get_device(device: str = "auto") -> torch.device:
    """
    Get the device for training.
    
    Args:
        device: Device string ('auto', 'cuda', 'cpu')
    
    Returns:
        torch.device object
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    device = torch.device(device)
    print(f"✓ Using device: {device}")
    
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def create_masks(
    src: torch.Tensor,
    tgt: torch.Tensor,
    src_pad_idx: int = 0,
    tgt_pad_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create source and target masks.
    
    Args:
        src: Source tensor of shape (batch, src_len)
        tgt: Target tensor of shape (batch, tgt_len)
        src_pad_idx: Source padding token index
        tgt_pad_idx: Target padding token index
    
    Returns:
        Tuple of (src_mask, tgt_mask)
    """
    # Source padding mask: (batch, 1, 1, src_len)
    src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2)
    
    # Target masks
    batch_size, tgt_len = tgt.size()
    
    # Padding mask: (batch, 1, 1, tgt_len)
    tgt_padding_mask = (tgt != tgt_pad_idx).unsqueeze(1).unsqueeze(2)
    
    # Causal mask: (1, 1, tgt_len, tgt_len)
    causal_mask = torch.tril(
        torch.ones(tgt_len, tgt_len, device=tgt.device)
    ).unsqueeze(0).unsqueeze(0).bool()
    
    # Combined target mask
    tgt_mask = tgt_padding_mask & causal_mask
    
    return src_mask, tgt_mask


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        path: Save path
        **kwargs: Additional items to save
    """
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    torch.save(checkpoint, path)
    print(f"✓ Saved checkpoint: {path}")


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu"
) -> dict:
    """
    Load training checkpoint.
    
    Args:
        path: Checkpoint path
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load to
    
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Loaded checkpoint: {path} (epoch {checkpoint.get('epoch', '?')})")
    return checkpoint
