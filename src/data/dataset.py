"""
Translation Dataset.
Loads data from Hugging Face Hub or local files.
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, List, Callable, Union
from pathlib import Path


class LocalTranslationDataset(Dataset):
    """
    PyTorch Dataset for translation task.
    Loads data from local parallel text files (.en, .vi).
    
    This is suitable for datasets like PhoMT from VinAI.
    """
    
    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        tokenizer_src: Optional[Callable] = None,
        tokenizer_tgt: Optional[Callable] = None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            src_file: Path to source language file (e.g., train.en)
            tgt_file: Path to target language file (e.g., train.vi)
            tokenizer_src: Source tokenizer function
            tokenizer_tgt: Target tokenizer function
            max_samples: Maximum number of samples to load (None = all)
        """
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        
        # Load parallel texts
        print(f"Loading data from local files...")
        print(f"  Source: {src_file}")
        print(f"  Target: {tgt_file}")
        
        self.src_texts = self._load_file(src_file, max_samples)
        self.tgt_texts = self._load_file(tgt_file, max_samples)
        
        assert len(self.src_texts) == len(self.tgt_texts), \
            f"Mismatch: {len(self.src_texts)} src vs {len(self.tgt_texts)} tgt"
        
        print(f"✓ Loaded {len(self.src_texts)} samples")
    
    def _load_file(self, file_path: str, max_samples: Optional[int] = None) -> List[str]:
        """Load text file, one sentence per line."""
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                texts.append(line.strip())
        return texts
    
    def __len__(self) -> int:
        return len(self.src_texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, List[int]]]:
        """Get a single sample."""
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        result = {
            'src_text': src_text,
            'tgt_text': tgt_text,
        }
        
        # Tokenize if tokenizers are provided
        if self.tokenizer_src is not None:
            result['src_tokens'] = self.tokenizer_src(src_text)
        if self.tokenizer_tgt is not None:
            result['tgt_tokens'] = self.tokenizer_tgt(tgt_text)
        
        return result
    
    def get_all_texts(self) -> tuple:
        """Return all source and target texts (for tokenizer training)."""
        return self.src_texts, self.tgt_texts
    
    @staticmethod
    def load_texts(src_file: str, tgt_file: str, max_samples: Optional[int] = None) -> tuple:
        """
        Static method to load texts from files without creating a dataset instance.
        """
        texts_src = []
        texts_tgt = []
        
        with open(src_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                texts_src.append(line.strip())
                
        with open(tgt_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                texts_tgt.append(line.strip())
                
        return texts_src, texts_tgt


class TranslationDataset(Dataset):
    """
    PyTorch Dataset for translation task.
    Loads data from Hugging Face Hub.
    Supports PhoMT and other translation datasets.
    """
    
    def __init__(
        self,
        dataset_name: str = "vinai/PhoMT",
        dataset_config: Optional[str] = None,
        split: str = "train",
        src_lang: str = "en",
        tgt_lang: str = "vi",
        max_seq_len: int = 128,
        tokenizer_src: Optional[Callable] = None,
        tokenizer_tgt: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            dataset_name: Hugging Face dataset name (e.g., 'vinai/PhoMT')
            dataset_config: Dataset configuration (None for PhoMT)
            split: Data split (train, validation, test)
            src_lang: Source language code
            tgt_lang: Target language code
            max_seq_len: Maximum sequence length
            tokenizer_src: Source tokenizer function
            tokenizer_tgt: Target tokenizer function
            cache_dir: Cache directory for downloaded data
            token: Hugging Face token for gated datasets
            max_samples: Maximum number of samples to load (None = all)
        """
        from datasets import load_dataset
        
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_seq_len = max_seq_len
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.dataset_name = dataset_name
        
        # Load dataset from Hugging Face
        print(f"Loading dataset: {dataset_name} ({split})...")
        
        load_kwargs = {
            'path': dataset_name,
            'split': split,
            'cache_dir': cache_dir,
            'trust_remote_code': True,
        }
        
        if dataset_config:
            load_kwargs['name'] = dataset_config
        if token:
            load_kwargs['token'] = token
        
        self.dataset = load_dataset(**load_kwargs)
        
        # Limit samples if specified
        if max_samples is not None and max_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(max_samples))
        
        print(f"✓ Loaded {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - src_tokens: Source token IDs
                - tgt_tokens: Target token IDs
                - src_text: Original source text
                - tgt_text: Original target text
        """
        item = self.dataset[idx]
        
        # Handle different dataset formats
        if 'translation' in item:
            src_text = item['translation'][self.src_lang]
            tgt_text = item['translation'][self.tgt_lang]
        else:
            src_text = item.get(self.src_lang, item.get('source', ''))
            tgt_text = item.get(self.tgt_lang, item.get('target', ''))
        
        result = {
            'src_text': src_text,
            'tgt_text': tgt_text,
        }
        
        # Tokenize if tokenizers are provided
        if self.tokenizer_src is not None:
            result['src_tokens'] = self.tokenizer_src(src_text)
        if self.tokenizer_tgt is not None:
            result['tgt_tokens'] = self.tokenizer_tgt(tgt_text)
        
        return result


class SimpleTokenizer:
    """
    Simple word-level tokenizer for demonstration.
    In production, use SentencePiece or similar.
    """
    
    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    
    PAD_IDX = 0
    UNK_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """Initialize with optional vocabulary."""
        if vocab is None:
            self.vocab = {
                self.PAD_TOKEN: self.PAD_IDX,
                self.UNK_TOKEN: self.UNK_IDX,
                self.BOS_TOKEN: self.BOS_IDX,
                self.EOS_TOKEN: self.EOS_IDX,
            }
            self.vocab_size = 4
        else:
            self.vocab = vocab
            self.vocab_size = len(vocab)
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def build_vocab(self, texts: List[str], max_vocab_size: int = 32000):
        """Build vocabulary from texts."""
        from collections import Counter
        
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Keep most common words
        most_common = word_counts.most_common(max_vocab_size - 4)
        
        for word, _ in most_common:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        
        self.vocab_size = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        print(f"✓ Built vocabulary with {self.vocab_size} tokens")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        words = text.lower().split()
        tokens = [self.vocab.get(w, self.UNK_IDX) for w in words]
        
        if add_special_tokens:
            tokens = [self.BOS_IDX] + tokens + [self.EOS_IDX]
        
        return tokens
    
    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        special = {self.PAD_IDX, self.UNK_IDX, self.BOS_IDX, self.EOS_IDX}
        words = []
        
        for t in tokens:
            if skip_special_tokens and t in special:
                continue
            word = self.inv_vocab.get(t, self.UNK_TOKEN)
            words.append(word)
        
        return ' '.join(words)
    
    def __call__(self, text: str) -> List[int]:
        """Tokenize text."""
        return self.encode(text)
    
    def save(self, path: str):
        """Save vocabulary to file."""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'SimpleTokenizer':
        """Load vocabulary from file."""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return cls(vocab)
