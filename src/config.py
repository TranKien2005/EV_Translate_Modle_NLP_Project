"""
Configuration loader.
Paths default to project-relative. Override from notebooks for Kaggle/Colab.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).parent
    # Go up from src/ to project root
    return current.parent


@dataclass
class PathConfig:
    """Path configuration - read from YAML config."""
    data_dir: Path = field(default_factory=lambda: Path("data/"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints/"))
    log_dir: Path = field(default_factory=lambda: Path("logs/"))
    
    def setup(self, project_root: Path, paths_config: Optional[Dict[str, Any]] = None):
        """Setup paths from YAML config.
        
        Args:
            project_root: Project root directory
            paths_config: 'paths' section from YAML config
        """
        # Get paths from config (or use defaults)
        data_dir = paths_config.get('data_dir', 'data') if paths_config else 'data'
        checkpoint_dir = paths_config.get('checkpoint_dir', 'checkpoints') if paths_config else 'checkpoints'
        log_dir = paths_config.get('log_dir', 'logs') if paths_config else 'logs'
        
        # Convert to Path - if absolute, use as-is; if relative, join with project_root
        self.data_dir = Path(data_dir) if Path(data_dir).is_absolute() else project_root / data_dir
        self.checkpoint_dir = Path(checkpoint_dir) if Path(checkpoint_dir).is_absolute() else project_root / checkpoint_dir
        self.log_dir = Path(log_dir) if Path(log_dir).is_absolute() else project_root / log_dir
    
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
            overrides: Dictionary of config overrides (including paths).
        """
        self.project_root = get_project_root()
        
        # Load base config
        if config_path is None:
            config_path = self.project_root / "config" / "config.yaml"
        
        self._config = self._load_yaml(config_path)
        
        # Apply overrides to config (including paths section)
        if overrides:
            self._apply_overrides(overrides)
        
        # Setup paths from YAML config 'paths' section
        paths_config = self._config.get('paths', {})
        # Merge any path overrides from overrides dict
        if overrides:
            for key in ['data_dir', 'checkpoint_dir', 'log_dir']:
                if key in overrides:
                    paths_config[key] = overrides[key]
        
        self.paths = PathConfig()
        self.paths.setup(self.project_root, paths_config)
        self.paths.create_dirs()
        
        # Parse config sections
        self._parse_config()
        
        print(f"✓ Config loaded | data_dir: {self.paths.data_dir}")
    
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
        """Parse config into attributes. All values must be in YAML config."""
        # Data config
        data = self._config.get('data', {})
        self.data_source = data['source']
        
        # Hugging Face dataset
        self.dataset_name = data.get('dataset_name')
        self.dataset_config = data.get('dataset_config')
        # Load HF token from environment variable (set in .env file)
        import os
        from dotenv import load_dotenv
        load_dotenv()
        self.hf_token = os.getenv('HF_TOKEN')
        
        # Local file paths (raw data) - relative to data_dir
        self.train_src = data.get('train_src', '')
        self.train_tgt = data.get('train_tgt', '')
        self.val_src = data.get('val_src', '')
        self.val_tgt = data.get('val_tgt', '')
        self.test_src = data.get('test_src', '')
        self.test_tgt = data.get('test_tgt', '')
        
        # Processed data paths - relative to data_dir
        self.processed_train = data.get('processed_train', '')
        self.processed_val = data.get('processed_val', '')
        self.processed_test = data.get('processed_test', '')
        
        # Evaluation limits
        self.eval_max_samples_bleu = data.get('eval_max_samples_bleu', 1000)
        self.eval_max_samples_gemini = data.get('eval_max_samples_gemini', 200)
        
        self.max_seq_len = data['max_seq_len']
        self.min_seq_len = data.get('min_seq_len', 1)
        self.max_samples = data.get('max_samples')
        self.src_vocab_size = data['src_vocab_size']
        self.tgt_vocab_size = data['tgt_vocab_size']
        
        # Model config - ALL REQUIRED
        model = self._config['model']
        self.d_model = model['d_model']
        self.num_heads = model['num_heads']
        self.num_encoder_layers = model['num_encoder_layers']
        self.num_decoder_layers = model['num_decoder_layers']
        self.d_ff = model['d_ff']
        self.dropout = model['dropout']
        
        # Training config - ALL REQUIRED
        training = self._config['training']
        self.batch_size = training['batch_size']
        self.gradient_accumulation_steps = training['gradient_accumulation_steps']
        self.epochs = training['epochs']
        self.learning_rate = training['learning_rate']
        self.warmup_steps = training['warmup_steps']
        self.label_smoothing = training['label_smoothing']
        self.gradient_clip = training['gradient_clip']
        self.weight_decay = training['weight_decay']
        self.min_lr = training['min_lr']
        
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
