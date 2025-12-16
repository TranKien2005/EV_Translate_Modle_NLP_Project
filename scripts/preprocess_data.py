"""
Preprocess and cache translation data.

This script:
1. Loads raw parallel text files
2. Trains SentencePiece tokenizers (if not exists)
3. Tokenizes all sentences
4. Filters by min/max token length
5. Saves processed data to .pt files for fast loading

Usage:
    python scripts/preprocess_data.py
    
    # With custom max samples
    python scripts/preprocess_data.py --max-samples 100000
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.data.tokenizer import SentencePieceTokenizer, train_tokenizers


def load_parallel_texts(src_file: str, tgt_file: str, max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """Load parallel text files."""
    print(f"Loading parallel texts...")
    print(f"  Source: {src_file}")
    print(f"  Target: {tgt_file}")
    
    src_texts = []
    tgt_texts = []
    
    with open(src_file, 'r', encoding='utf-8') as f_src, \
         open(tgt_file, 'r', encoding='utf-8') as f_tgt:
        for i, (src_line, tgt_line) in enumerate(zip(f_src, f_tgt)):
            if max_samples is not None and i >= max_samples:
                break
            src_texts.append(src_line.strip())
            tgt_texts.append(tgt_line.strip())
    
    print(f"  ✓ Loaded {len(src_texts)} sentence pairs")
    return src_texts, tgt_texts


def tokenize_data(
    src_texts: List[str],
    tgt_texts: List[str],
    tokenizer_src: SentencePieceTokenizer,
    tokenizer_tgt: SentencePieceTokenizer,
    min_len: int = 1,
    max_len: int = 128,
    show_progress: bool = True
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Tokenize and filter data by length.
    
    Args:
        src_texts: Source sentences
        tgt_texts: Target sentences
        tokenizer_src: Source tokenizer
        tokenizer_tgt: Target tokenizer
        min_len: Minimum token length (inclusive)
        max_len: Maximum token length (inclusive)
        show_progress: Show progress bar
    
    Returns:
        Tuple of (src_tokens_list, tgt_tokens_list)
    """
    src_tokens_all = []
    tgt_tokens_all = []
    
    filtered_count = 0
    
    iterator = zip(src_texts, tgt_texts)
    if show_progress:
        iterator = tqdm(iterator, total=len(src_texts), desc="Tokenizing")
    
    for src_text, tgt_text in iterator:
        # Skip empty sentences
        if not src_text or not tgt_text:
            filtered_count += 1
            continue
        
        # Tokenize (with BOS/EOS)
        src_tokens = tokenizer_src.encode(src_text, add_bos=True, add_eos=True)
        tgt_tokens = tokenizer_tgt.encode(tgt_text, add_bos=True, add_eos=True)
        
        # Check length constraints
        if len(src_tokens) < min_len or len(src_tokens) > max_len:
            filtered_count += 1
            continue
        if len(tgt_tokens) < min_len or len(tgt_tokens) > max_len:
            filtered_count += 1
            continue
        
        src_tokens_all.append(src_tokens)
        tgt_tokens_all.append(tgt_tokens)
    
    print(f"  ✓ Kept {len(src_tokens_all)} pairs, filtered {filtered_count}")
    return src_tokens_all, tgt_tokens_all


def save_processed_data(
    src_tokens: List[List[int]],
    tgt_tokens: List[List[int]],
    output_path: str
):
    """Save processed data to .pt file."""
    data = {
        'src': src_tokens,
        'tgt': tgt_tokens,
        'num_samples': len(src_tokens)
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(data, output_path)
    
    # Calculate file size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved to {output_path} ({size_mb:.1f} MB)")


def preprocess_split(
    src_file: str,
    tgt_file: str,
    output_path: str,
    tokenizer_src: SentencePieceTokenizer,
    tokenizer_tgt: SentencePieceTokenizer,
    min_len: int = 1,
    max_len: int = 128,
    max_samples: Optional[int] = None
):
    """Preprocess a single data split."""
    print(f"\n{'='*60}")
    print(f"Processing: {Path(output_path).stem}")
    print(f"{'='*60}")
    
    # Load raw texts
    src_texts, tgt_texts = load_parallel_texts(src_file, tgt_file, max_samples)
    
    # Tokenize and filter
    src_tokens, tgt_tokens = tokenize_data(
        src_texts, tgt_texts,
        tokenizer_src, tokenizer_tgt,
        min_len=min_len, max_len=max_len
    )
    
    # Save
    save_processed_data(src_tokens, tgt_tokens, output_path)
    
    return len(src_tokens)


def main():
    parser = argparse.ArgumentParser(description="Preprocess translation data")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to process (None = all)")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Force retrain tokenizers even if they exist")
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    print("="*60)
    print("Data Preprocessing Script")
    print("="*60)
    print(f"Config loaded | Env: {config.env}")
    
    # Setup paths
    project_root = config.project_root
    train_src = project_root / config.train_src
    train_tgt = project_root / config.train_tgt
    val_src = project_root / config.val_src
    val_tgt = project_root / config.val_tgt
    
    # Check if raw data exists
    if not train_src.exists():
        print(f"\n❌ Raw data not found: {train_src}")
        print("Please run: python scripts/download_phomt.py first")
        sys.exit(1)
    
    # Tokenizer paths
    tokenizer_dir = config.paths.checkpoint_dir / "tokenizers"
    tokenizer_src_path = tokenizer_dir / config.tokenizer_src_file
    tokenizer_tgt_path = tokenizer_dir / config.tokenizer_tgt_file
    
    # Train or load tokenizers
    if tokenizer_src_path.exists() and not args.force_retrain:
        print(f"\n✓ Loading existing tokenizers from {tokenizer_dir}")
        tokenizer_src = SentencePieceTokenizer(str(tokenizer_src_path))
        tokenizer_tgt = SentencePieceTokenizer(str(tokenizer_tgt_path))
    else:
        print(f"\nTraining new tokenizers...")
        
        # Load texts for training
        src_texts, tgt_texts = load_parallel_texts(
            str(train_src), str(train_tgt), 
            max_samples=args.max_samples or config.max_samples
        )
        
        # Train tokenizers
        tokenizer_src, tokenizer_tgt = train_tokenizers(
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            output_dir=str(tokenizer_dir),
            src_vocab_size=config.src_vocab_size,
            tgt_vocab_size=config.tgt_vocab_size,
            model_type="bpe"
        )
    
    # Output paths for processed data
    processed_dir = config.paths.data_dir / "processed"
    train_output = processed_dir / "train.pt"
    val_output = processed_dir / "val.pt"
    
    # Process training data
    train_count = preprocess_split(
        src_file=str(train_src),
        tgt_file=str(train_tgt),
        output_path=str(train_output),
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        min_len=config._config.get('data', {}).get('min_seq_len', 1),
        max_len=config.max_seq_len,
        max_samples=args.max_samples or config.max_samples
    )
    
    # Process validation data
    val_count = preprocess_split(
        src_file=str(val_src),
        tgt_file=str(val_tgt),
        output_path=str(val_output),
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        min_len=config.min_seq_len,
        max_len=config.max_seq_len,
        max_samples=None
    )
    
    # Process test data (if exists)
    test_src = project_root / config.test_src
    test_tgt = project_root / config.test_tgt
    test_output = config.paths.data_dir / "processed" / "test.pt"
    
    if test_src.exists() and test_tgt.exists():
        test_count = preprocess_split(
            src_file=str(test_src),
            tgt_file=str(test_tgt),
            output_path=str(test_output),
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
            min_len=config.min_seq_len,
            max_len=config.max_seq_len,
            max_samples=None
        )
        print(f"Test samples:  {test_count:,}")
    else:
        print("\n⚠ Test files not found, skipping test set processing.")
        test_count = 0
    
    # Summary
    print(f"\n{'='*60}")
    print("Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"Train samples: {train_count:,}")
    print(f"Val samples:   {val_count:,}")
    if test_count > 0:
        print(f"Test samples:  {test_count:,}")
    print(f"\nProcessed data saved to: {processed_dir}")
    print(f"Tokenizers saved to: {tokenizer_dir}")


if __name__ == "__main__":
    main()
