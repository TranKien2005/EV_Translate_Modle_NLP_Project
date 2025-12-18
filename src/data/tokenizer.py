"""
SentencePiece Tokenizer for Translation.
Supports training new models and loading pre-trained models.
"""

import os
from pathlib import Path
from typing import List, Optional, Union
import sentencepiece as spm


class SentencePieceTokenizer:
    """
    SentencePiece tokenizer wrapper for translation tasks.
    
    Supports:
    - Training new SentencePiece models
    - Loading pre-trained models
    - Encoding/Decoding text
    """
    
    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    
    PAD_IDX = 0
    UNK_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize tokenizer.
        
        Args:
            model_path: Path to .model file (if loading pre-trained)
        """
        self.sp = spm.SentencePieceProcessor()
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    @classmethod
    def train(
        cls,
        input_file: str,
        model_prefix: str,
        vocab_size: int = 32000,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
        pad_id: int = 0,
        unk_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
        **kwargs
    ) -> 'SentencePieceTokenizer':
        """
        Train a new SentencePiece model.
        
        Args:
            input_file: Path to training text file (one sentence per line)
            model_prefix: Prefix for output model files (.model, .vocab)
            vocab_size: Vocabulary size
            model_type: Model type ('bpe', 'unigram', 'char', 'word')
            character_coverage: Character coverage (1.0 for English, 0.9995 for Vietnamese)
            pad_id, unk_id, bos_id, eos_id: Special token IDs
            **kwargs: Additional SentencePiece training arguments
        
        Returns:
            Trained SentencePieceTokenizer instance
        """
        # Create output directory if needed
        output_dir = Path(model_prefix).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build training command
        train_args = {
            'input': input_file,
            'model_prefix': model_prefix,
            'vocab_size': vocab_size,
            'model_type': model_type,
            'character_coverage': character_coverage,
            'pad_id': pad_id,
            'unk_id': unk_id,
            'bos_id': bos_id,
            'eos_id': eos_id,
            'pad_piece': cls.PAD_TOKEN,
            'unk_piece': cls.UNK_TOKEN,
            'bos_piece': cls.BOS_TOKEN,
            'eos_piece': cls.EOS_TOKEN,
            **kwargs
        }
        
        # Convert to command string
        cmd = ' '.join([f'--{k}={v}' for k, v in train_args.items()])
        
        print(f"Training SentencePiece model...")
        print(f"  Input: {input_file}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Model type: {model_type}")
        
        spm.SentencePieceTrainer.Train(cmd)
        
        print(f"✓ Model saved to: {model_prefix}.model")
        
        # Load and return the trained model
        return cls(f"{model_prefix}.model")
    
    def load(self, model_path: str):
        """Load a pre-trained SentencePiece model."""
        self.model_path = model_path
        self.sp.Load(model_path)
        print(f"✓ Loaded SentencePiece model: {model_path}")
        print(f"  Vocab size: {self.vocab_size}")
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.sp.GetPieceSize()
    
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_bos: Whether to add BOS token
            add_eos: Whether to add EOS token
        
        Returns:
            List of token IDs
        """
        tokens = self.sp.EncodeAsIds(text)
        
        if add_bos:
            tokens = [self.BOS_IDX] + tokens
        if add_eos:
            tokens = tokens + [self.EOS_IDX]
        
        return tokens
    
    def decode(
        self,
        tokens: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            tokens: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        if skip_special_tokens:
            special = {self.PAD_IDX, self.UNK_IDX, self.BOS_IDX, self.EOS_IDX}
            tokens = [t for t in tokens if t not in special]
        
        return self.sp.DecodeIds(tokens)
    
    def encode_as_pieces(self, text: str) -> List[str]:
        """Encode text to subword pieces."""
        return self.sp.EncodeAsPieces(text)
    
    def __call__(self, text: str) -> List[int]:
        """
        Tokenize text (with BOS and EOS).
        
        Args:
            text: Input text
        
        Returns:
            List of token IDs
        """
        return self.encode(text, add_bos=True, add_eos=True)
    
    def batch_encode(
        self,
        texts: List[str],
        add_bos: bool = True,
        add_eos: bool = True
    ) -> List[List[int]]:
        """Encode a batch of texts."""
        return [self.encode(text, add_bos, add_eos) for text in texts]
    
    def batch_decode(
        self,
        batch_tokens: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode a batch of token sequences."""
        return [self.decode(tokens, skip_special_tokens) for tokens in batch_tokens]


def prepare_training_data(
    texts: List[str],
    output_path: str
) -> str:
    """
    Prepare training data file for SentencePiece.
    
    Args:
        texts: List of text sentences
        output_path: Output file path
    
    Returns:
        Path to created file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            # Clean and write
            text = text.strip()
            if text:
                f.write(text + '\n')
    
    print(f"✓ Saved {len(texts)} sentences to {output_path}")
    return str(output_path)


def train_tokenizers(
    src_texts: List[str],
    tgt_texts: List[str],
    output_dir: str,
    src_vocab_size: int = 32000,
    tgt_vocab_size: int = 32000,
    model_type: str = "bpe",
    src_model_prefix: str = "tokenizer_src",
    tgt_model_prefix: str = "tokenizer_tgt"
) -> tuple:
    """
    Train source and target tokenizers.
    
    Args:
        src_texts: Source language texts
        tgt_texts: Target language texts
        output_dir: Output directory for models
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        model_type: SentencePiece model type
        src_model_prefix: Prefix for source tokenizer (default: tokenizer_src)
        tgt_model_prefix: Prefix for target tokenizer (default: tokenizer_tgt)
    
    Returns:
        Tuple of (src_tokenizer, tgt_tokenizer)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare training data
    print("Preparing training data...")
    src_file = prepare_training_data(src_texts, output_dir / "src_train.txt")
    tgt_file = prepare_training_data(tgt_texts, output_dir / "tgt_train.txt")
    
    # Train source tokenizer
    print(f"\nTraining source tokenizer ({src_model_prefix})...")
    src_tokenizer = SentencePieceTokenizer.train(
        input_file=src_file,
        model_prefix=str(output_dir / src_model_prefix),
        vocab_size=src_vocab_size,
        model_type=model_type,
        character_coverage=1.0  # Default for English
    )
    
    # Train target tokenizer
    print(f"\nTraining target tokenizer ({tgt_model_prefix})...")
    tgt_tokenizer = SentencePieceTokenizer.train(
        input_file=tgt_file,
        model_prefix=str(output_dir / tgt_model_prefix),
        vocab_size=tgt_vocab_size,
        model_type=model_type,
        character_coverage=0.9995  # More characters for non-English
    )
    
    print("\n✓ Tokenizers trained successfully!")
    print(f"  Source vocab: {src_tokenizer.vocab_size}")
    print(f"  Target vocab: {tgt_tokenizer.vocab_size}")
    
    return src_tokenizer, tgt_tokenizer
