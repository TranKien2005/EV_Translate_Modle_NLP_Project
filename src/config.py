"""
Configuration loader with environment auto-detection.
Supports: Local, Kaggle, Google Colab
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


def detect_environment() -> str:
    """Detect the current running environment."""
    if os.path.exists('/kaggle'):
        return 'kaggle'
    elif os.path.exists('/content'):
        return 'colab'
    return 'local'


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).parent
    # Go up from src/ to project root
    return current.parent


@dataclass
class PathConfig:
    """Path configuration with environment-aware defaults."""
    data_dir: Path = field(default_factory=lambda: Path("data/"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/"))
    log_dir: Path = field(default_factory=lambda: Path("logs/"))
    
    def setup_for_environment(self, env: str, project_root: Path):
        """Setup paths based on environment."""
        if env == 'kaggle':
            # Kaggle-specific paths
            self.data_dir = Path('/kaggle/input/')
            self.checkpoint_dir = Path('/kaggle/working/checkpoints/')
            self.log_dir = Path('/kaggle/working/logs/')
        elif env == 'colab':
            # Colab-specific paths
            self.data_dir = Path('/content/drive/MyDrive/data/')
            self.checkpoint_dir = Path('/content/checkpoints/')
            self.log_dir = Path('/content/logs/')
        else:
            # Local paths (relative to project root)
            self.data_dir = project_root / "data"
            self.checkpoint_dir = project_root / "checkpoints"
            self.log_dir = project_root / "logs"
    
    def create_dirs(self):
        """Create directories if they don't exist."""
        for path in [self.checkpoint_dir, self.log_dir]:
            path.mkdir(parents=True, exist_ok=True)


class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file. If None, uses default.
            overrides: Dictionary of config overrides.
        """
        self.env = detect_environment()
        self.project_root = get_project_root()
        
        # Load base config
        if config_path is None:
            config_path = self.project_root / "config" / "config.yaml"
        
        self._config = self._load_yaml(config_path)
        
        # Apply overrides
        if overrides:
            self._apply_overrides(overrides)
        
        # Setup paths
        self.paths = PathConfig()
        self.paths.setup_for_environment(self.env, self.project_root)
        self.paths.create_dirs()
        
        # Parse config sections
        self._parse_config()
        
        print(f"✓ Config loaded | Environment: {self.env}")
    
    def _load_yaml(self, path: Path) -> dict:
        """Load YAML config file."""
        path = Path(path)
        if not path.exists():
            print(f"⚠ Config file not found: {path}, using defaults")
            return {}
        
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _apply_overrides(self, overrides: Dict[str, Any]):
        """Apply config overrides."""
        def deep_update(base: dict, updates: dict):
            for key, value in updates.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value
        
        deep_update(self._config, overrides)
    
    def _parse_config(self):
        """Parse config into attributes."""
        # Data config
        data = self._config.get('data', {})
        self.data_source = data.get('source', 'local')  # 'local', 'huggingface', or 'processed'
        
        # Hugging Face dataset
        self.dataset_name = data.get('dataset_name', 'vinai/PhoMT')
        self.dataset_config = data.get('dataset_config', None)
        # Load HF token from environment variable (set in .env file)
        import os
        from dotenv import load_dotenv
        load_dotenv()
        self.hf_token = os.getenv('HF_TOKEN', None)
        
        # Local file paths (raw data)
        self.train_src = data.get('train_src', '')
        self.train_tgt = data.get('train_tgt', '')
        self.val_src = data.get('val_src', '')
        self.val_tgt = data.get('val_tgt', '')
        self.test_src = data.get('test_src', '')
        self.test_tgt = data.get('test_tgt', '')
        
        # Processed data paths
        self.processed_train = data.get('processed_train', 'data/processed/train.pt')
        self.processed_val = data.get('processed_val', 'data/processed/val.pt')
        self.processed_test = data.get('processed_test', 'data/processed/test.pt')
        
        # Evaluation limits
        self.eval_max_samples_bleu = data.get('eval_max_samples_bleu', 1000)
        self.eval_max_samples_gemini = data.get('eval_max_samples_gemini', 200)
        
        self.max_seq_len = data.get('max_seq_len', 128)
        self.min_seq_len = data.get('min_seq_len', 1)
        self.max_samples = data.get('max_samples', None)
        self.src_vocab_size = data.get('src_vocab_size', 32000)
        self.tgt_vocab_size = data.get('tgt_vocab_size', 32000)
        
        # Model config
        model = self._config.get('model', {})
        self.d_model = model.get('d_model', 512)
        self.num_heads = model.get('num_heads', 8)
        self.num_encoder_layers = model.get('num_encoder_layers', 6)
        self.num_decoder_layers = model.get('num_decoder_layers', 6)
        self.d_ff = model.get('d_ff', 2048)
        self.dropout = model.get('dropout', 0.1)
        
        # Training config
        training = self._config.get('training', {})
        self.batch_size = training.get('batch_size', 32)
        self.gradient_accumulation_steps = training.get('gradient_accumulation_steps', 1)
        self.epochs = training.get('epochs', 50)
        self.learning_rate = training.get('learning_rate', 1e-4)
        self.warmup_steps = training.get('warmup_steps', 4000)
        self.label_smoothing = training.get('label_smoothing', 0.1)
        self.gradient_clip = training.get('gradient_clip', 1.0)
        
        # Output files config
        output_files = self._config.get('output_files', {})
        self.tokenizer_src_file = output_files.get('tokenizer_src', 'tokenizer_src.model')
        self.tokenizer_tgt_file = output_files.get('tokenizer_tgt', 'tokenizer_tgt.model')
        self.vocab_src_file = output_files.get('vocab_src', 'tokenizer_src.vocab')
        self.vocab_tgt_file = output_files.get('vocab_tgt', 'tokenizer_tgt.vocab')
        self.best_model_file = output_files.get('best_model', 'best_model.pt')
        self.last_model_file = output_files.get('last_model', 'last_model.pt')
        self.checkpoint_pattern = output_files.get('checkpoint_pattern', 'checkpoint_epoch_{epoch}.pt')
        
        # Hardware
        hardware = self._config.get('hardware', {})
        device_setting = hardware.get('device', 'auto')
        if device_setting == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_setting
        self.num_workers = hardware.get('num_workers', 4)
    
    # Helper methods to get full paths
    def get_tokenizer_src_path(self) -> Path:
        """Get full path to source tokenizer model."""
        return self.paths.checkpoint_dir / "tokenizers" / self.tokenizer_src_file
    
    def get_tokenizer_tgt_path(self) -> Path:
        """Get full path to target tokenizer model."""
        return self.paths.checkpoint_dir / "tokenizers" / self.tokenizer_tgt_file
    
    def get_best_model_path(self) -> Path:
        """Get full path to best model checkpoint."""
        return self.paths.checkpoint_dir / self.best_model_file
    
    def get_checkpoint_path(self, epoch: int) -> Path:
        """Get full path to epoch checkpoint."""
        filename = self.checkpoint_pattern.format(epoch=epoch)
        return self.paths.checkpoint_dir / filename
    
    def __repr__(self):
        return f"Config(env={self.env}, device={self.device}, d_model={self.d_model})"


# Convenience function
def load_config(config_path: Optional[str] = None, **overrides) -> Config:
    """Load configuration with optional overrides."""
    return Config(config_path, overrides if overrides else None)
