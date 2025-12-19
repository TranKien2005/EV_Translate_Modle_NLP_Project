"""
Script to download and extract PhoMT dataset from Hugging Face.

Usage:
    python scripts/download_phomt.py
    
    # Or with custom token
    python scripts/download_phomt.py --token YOUR_HF_TOKEN
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download


# Default paths
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "PhoMT"
DATASET_REPO = "vinai/PhoMT"
DATASET_FILE = "PhoMT.zip"


def download_phomt(
    output_dir: Path,
    token: str = None,
    force: bool = False
) -> Path:
    """
    Download and extract PhoMT dataset.
    
    Args:
        output_dir: Directory to extract data to
        token: Hugging Face token for authentication
        force: Force re-download even if exists
    
    Returns:
        Path to extracted data directory
    """
    output_dir = Path(output_dir)
    
    # Check if already extracted
    train_en = output_dir / "detokenization" / "train" / "train.en"
    if train_en.exists() and not force:
        print(f"✓ PhoMT already exists at: {output_dir}")
        return output_dir
    
    print("="*60)
    print("Downloading PhoMT Dataset from Hugging Face")
    print("="*60)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download zip file to a temp location
    print(f"\n1. Downloading {DATASET_FILE}...")
    print("   (This may take a while, ~500MB)")
    
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        zip_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=DATASET_FILE,
            repo_type="dataset",
            token=token,
            local_dir=str(temp_dir),
            local_dir_use_symlinks=False
        )
        print(f"   ✓ Downloaded to: {zip_path}")
    except Exception as e:
        print(f"\n❌ Error downloading: {e}")
        print("\nPossible solutions:")
        print("1. Make sure you've accepted the dataset terms at:")
        print("   https://huggingface.co/datasets/vinai/PhoMT")
        print("2. Provide a valid Hugging Face token with --token")
        sys.exit(1)
    
    # Extract zip file to temp, then move to correct location
    print(f"\n2. Extracting {DATASET_FILE}...")
    try:
        extract_temp = temp_dir / "extracted"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_temp)
        
        # Find where detokenization folder is
        detok_path = None
        for root, dirs, files in os.walk(extract_temp):
            if 'detokenization' in dirs:
                detok_path = Path(root) / 'detokenization'
                break
        
        if detok_path is None:
            print("   ❌ Could not find 'detokenization' folder in zip")
            sys.exit(1)
        
        # Move detokenization to output_dir
        import shutil
        target_detok = output_dir / "detokenization"
        if target_detok.exists():
            shutil.rmtree(target_detok)
        shutil.move(str(detok_path), str(target_detok))
        
        print(f"   ✓ Extracted to: {output_dir}")
        
        # Cleanup temp
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"   ❌ Error extracting: {e}")
        sys.exit(1)
    
    # Verify extraction
    print("\n3. Verifying extracted files...")
    expected_files = [
        ("detokenization/train/train.en", "Training EN"),
        ("detokenization/train/train.vi", "Training VI"),
        ("detokenization/dev/dev.en", "Dev EN"),
        ("detokenization/dev/dev.vi", "Dev VI"),
        ("detokenization/test/test.en", "Test EN"),
        ("detokenization/test/test.vi", "Test VI"),
    ]
    
    all_found = True
    for file_path, description in expected_files:
        full_path = output_dir / file_path
        if full_path.exists():
            # Count lines
            with open(full_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            print(f"   ✓ {description}: {line_count:,} lines")
        else:
            print(f"   ❌ {description}: NOT FOUND at {full_path}")
            all_found = False
    
    if all_found:
        print(f"\n✓ PhoMT dataset ready at: {output_dir}")
    else:
        print("\n⚠ Some files are missing!")
    
    print("="*60)
    
    return output_dir


def get_phomt_paths(base_dir: Path = None) -> dict:
    """
    Get paths to PhoMT data files.
    
    Args:
        base_dir: Base directory of PhoMT data
    
    Returns:
        Dictionary with paths to train/dev/test files
    """
    if base_dir is None:
        base_dir = DEFAULT_OUTPUT_DIR
    
    base_dir = Path(base_dir)
    
    # PhoMT has two versions: tokenization and detokenization
    # We use detokenization (raw text, not pre-tokenized)
    data_dir = base_dir / "detokenization"
    
    return {
        'train_src': str(data_dir / "train" / "train.en"),
        'train_tgt': str(data_dir / "train" / "train.vi"),
        'val_src': str(data_dir / "dev" / "dev.en"),
        'val_tgt': str(data_dir / "dev" / "dev.vi"),
        'test_src': str(data_dir / "test" / "test.en"),
        'test_tgt': str(data_dir / "test" / "test.vi"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PhoMT dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,  # Will be set from config if not provided
        help="Output directory for extracted data"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token for authentication"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data exists"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: config/config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Load config for token and paths
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.config import load_config
        config = load_config(args.config)
        
        # Use token from config if not provided
        if args.token is None:
            args.token = config.hf_token
            if args.token:
                print(f"Using token from .env")
        
        # Use output dir from config if not provided
        if args.output_dir is None:
            args.output_dir = str(config.paths.data_dir / "PhoMT")
            print(f"Using data dir from config: {args.output_dir}")
    except Exception as e:
        print(f"Could not load config: {e}")
        if args.output_dir is None:
            args.output_dir = str(DEFAULT_OUTPUT_DIR)
    
    output_dir = download_phomt(
        output_dir=Path(args.output_dir),
        token=args.token,
        force=args.force
    )
    
    # Print paths
    print("\nData paths for config:")
    paths = get_phomt_paths(output_dir)
    for key, value in paths.items():
        print(f"  {key}: {value}")
