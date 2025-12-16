"""
DataLoader utilities for translation task.
Includes collate function with dynamic padding.
"""

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Optional, Tuple

from .dataset import TranslationDataset


def collate_fn(
    batch: List[Dict[str, Any]],
    src_pad_idx: int = 0,
    tgt_pad_idx: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Collate function for translation batch.
    Pads sequences to the same length within each batch.
    
    Args:
        batch: List of samples from dataset
        src_pad_idx: Source padding token index
        tgt_pad_idx: Target padding token index
    
    Returns:
        Dictionary with padded tensors
    """
    # Check if tokenized
    has_tokens = 'src_tokens' in batch[0]
    
    if has_tokens:
        # Get sequences
        src_seqs = [torch.tensor(item['src_tokens']) for item in batch]
        tgt_seqs = [torch.tensor(item['tgt_tokens']) for item in batch]
        
        # Pad sequences
        src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=src_pad_idx)
        tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=tgt_pad_idx)
        
        result = {
            'src': src_padded,
            'tgt': tgt_padded,
        }
        
        # Add texts if available (not available for ProcessedDataset)
        if 'src_text' in batch[0]:
            result['src_texts'] = [item['src_text'] for item in batch]
            result['tgt_texts'] = [item['tgt_text'] for item in batch]
        
        return result
    else:
        # Return texts only (tokenization will be done later)
        return {
            'src_texts': [item['src_text'] for item in batch],
            'tgt_texts': [item['tgt_text'] for item in batch],
        }


def create_collate_fn(src_pad_idx: int = 0, tgt_pad_idx: int = 0):
    """Create a collate function with specified padding indices."""
    def _collate(batch):
        return collate_fn(batch, src_pad_idx, tgt_pad_idx)
    return _collate


def create_dataloaders(
    dataset_name: str = "mt_eng_vietnamese",
    dataset_config: str = "iwslt2015-en-vi",
    batch_size: int = 32,
    max_seq_len: int = 128,
    tokenizer_src=None,
    tokenizer_tgt=None,
    num_workers: int = 4,
    pin_memory: bool = True,
    src_pad_idx: int = 0,
    tgt_pad_idx: int = 0,
    cache_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        dataset_name: Hugging Face dataset name
        dataset_config: Dataset configuration
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        tokenizer_src: Source tokenizer
        tokenizer_tgt: Target tokenizer
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        src_pad_idx: Source padding token index
        tgt_pad_idx: Target padding token index
        cache_dir: Cache directory
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    collate = create_collate_fn(src_pad_idx, tgt_pad_idx)
    
    # Create datasets
    train_dataset = TranslationDataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split="train",
        max_seq_len=max_seq_len,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        cache_dir=cache_dir
    )
    
    val_dataset = TranslationDataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split="validation",
        max_seq_len=max_seq_len,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        cache_dir=cache_dir
    )
    
    # Try to load test set (not all datasets have it)
    try:
        test_dataset = TranslationDataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split="test",
            max_seq_len=max_seq_len,
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
            cache_dir=cache_dir
        )
    except Exception:
        test_dataset = None
        print("⚠ No test split found")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate
        )
    
    print(f"✓ Created DataLoaders:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    if test_loader:
        print(f"  Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader
