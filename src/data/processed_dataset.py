"""
Processed Dataset - loads pre-tokenized data from .pt files.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
from pathlib import Path


class ProcessedDataset(Dataset):
    """
    Dataset that loads pre-processed tokenized data from .pt files.
    Much faster than tokenizing on-the-fly.
    """
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to processed .pt file
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Processed data not found: {data_path}\n"
                "Please run: python scripts/preprocess_data.py first"
            )
        
        print(f"Loading processed data from: {data_path}")
        data = torch.load(data_path)
        
        # Store as lists (will convert to tensor in __getitem__)
        self.src_tokens = data['src']
        self.tgt_tokens = data['tgt']
        self.num_samples = data['num_samples']
        
        print(f"✓ Loaded {self.num_samples:,} samples")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample as tensors (faster for DataLoader)."""
        return {
            'src_tokens': torch.tensor(self.src_tokens[idx], dtype=torch.long),
            'tgt_tokens': torch.tensor(self.tgt_tokens[idx], dtype=torch.long),
        }


def load_processed_datasets(
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None
) -> tuple:
    """
    Load all processed datasets.
    
    Args:
        train_path: Path to processed training data
        val_path: Path to processed validation data
        test_path: Optional path to processed test data
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = ProcessedDataset(train_path)
    val_dataset = ProcessedDataset(val_path)
    
    test_dataset = None
    if test_path and Path(test_path).exists():
        test_dataset = ProcessedDataset(test_path)
    
    return train_dataset, val_dataset, test_dataset
