"""
Utility functions for data processing, training, and evaluation
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
from typing import List, Tuple


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, loss, path, step=0):
    # Tạo dictionary chứa mọi thứ cần thiết để resume
    checkpoint = {
        'epoch': epoch,
        'step': step,             # <--- Mới thêm: lưu lại đang ở step bao nhiêu
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, filepath, device):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")
    return epoch, loss


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss
    Reduces overconfidence in predictions
    """
    def __init__(self, vocab_size, padding_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        
    def forward(self, pred, target):
        """
        Args:
            pred: [batch_size * seq_len, vocab_size] (log probabilities)
            target: [batch_size * seq_len]
        """
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target == self.padding_idx, as_tuple=False)
            if mask.dim() > 0 and mask.size(0) > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
        
        return self.criterion(pred, true_dist)


class NoamScheduler:
    """
    Learning rate scheduler with warmup (Noam scheme)
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self):
        """Update learning rate"""
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def _get_lr(self):
        """Calculate learning rate"""
        step = max(1, self.step_num)
        return self.d_model ** (-0.5) * min(
            step ** (-0.5),
            step * self.warmup_steps ** (-1.5)
        )


def calculate_perplexity(loss):
    """Calculate perplexity from cross-entropy loss"""
    return np.exp(loss)


def get_device():
    """Get available device (GPU/CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def format_time(seconds):
    """Format seconds to readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_model_summary(model):
    """Print model architecture summary"""
    print("=" * 80)
    print("Model Architecture Summary")
    print("=" * 80)
    
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1e6:.2f} MB (assuming float32)")
    
    print("\nLayer-wise parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name:50s} {param.numel():>12,}")
    
    print("=" * 80)


class AverageMeter:
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_padding_mask(seq, pad_idx):
    """
    Create padding mask
    Args:
        seq: [batch_size, seq_len]
        pad_idx: Padding token index
    Returns:
        mask: [batch_size, 1, 1, seq_len]
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(size, device):
    """
    Create look-ahead mask for decoder
    Args:
        size: Sequence length
        device: torch device
    Returns:
        mask: [1, 1, size, size]
    """
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return ~mask


def create_masks(src, tgt, pad_idx, device):
    """
    Create all masks for training
    Args:
        src: Source sequence [batch_size, src_len]
        tgt: Target sequence [batch_size, tgt_len]
        pad_idx: Padding token index
        device: torch device
    Returns:
        src_mask: [batch_size, 1, 1, src_len]
        tgt_mask: [batch_size, 1, tgt_len, tgt_len]
    """
    src_mask = create_padding_mask(src, pad_idx)
    tgt_pad_mask = create_padding_mask(tgt, pad_idx)
    tgt_len = tgt.size(1)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_len, device)
    tgt_mask = tgt_pad_mask & tgt_look_ahead_mask.unsqueeze(0)
    return src_mask, tgt_mask
