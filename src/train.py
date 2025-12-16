"""
Training script for Transformer translation model.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm

from src.config import Config, load_config
from src.models import Transformer
from src.data import TranslationDataset, LocalTranslationDataset, ProcessedDataset
from src.data.dataloader import create_collate_fn
from src.data.tokenizer import SentencePieceTokenizer, train_tokenizers
from src.utils import set_seed, get_device, save_checkpoint, load_checkpoint
from src.utils.logger import TrainingLogger
from src.utils.metrics import MetricsTracker
from torch.utils.data import DataLoader


class Trainer:
    """
    Trainer class for Transformer model.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        config_path: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize trainer.
        
        Args:
            config: Config object (if None, will load from config_path)
            config_path: Path to config file
            overrides: Config overrides
        """
        # Load config
        if config is None:
            config = load_config(config_path, **(overrides or {}))
        self.config = config
        
        # Setup
        set_seed(42)
        self.device = get_device(config.device)
        
        # Logger
        self.logger = TrainingLogger(
            log_dir=str(config.paths.log_dir),
            use_tensorboard=True
        )
        
        # Model, optimizer, etc. will be initialized in setup()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.tokenizer_src = None
        self.tokenizer_tgt = None
        self.train_loader = None
        self.val_loader = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def load_tokenizers(self):
        """Load pre-trained tokenizers."""
        tokenizer_dir = self.config.paths.checkpoint_dir / "tokenizers"
        src_path = tokenizer_dir / self.config.tokenizer_src_file
        tgt_path = tokenizer_dir / self.config.tokenizer_tgt_file
        
        if not src_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {src_path}\n"
                "Please run: python scripts/preprocess_data.py first"
            )
        
        self.tokenizer_src = SentencePieceTokenizer(str(src_path))
        self.tokenizer_tgt = SentencePieceTokenizer(str(tgt_path))
        
        self.logger.log(f"✓ Loaded tokenizers from {tokenizer_dir}")
        self.logger.log(f"  Source vocab: {self.tokenizer_src.vocab_size}")
        self.logger.log(f"  Target vocab: {self.tokenizer_tgt.vocab_size}")
    
    def setup_tokenizers(self, train_texts_src: list, train_texts_tgt: list):
        """Setup SentencePiece tokenizers from training data."""
        self.logger.log("Training SentencePiece tokenizers...")
        
        # Train SentencePiece tokenizers
        tokenizer_dir = self.config.paths.checkpoint_dir / "tokenizers"
        self.tokenizer_src, self.tokenizer_tgt = train_tokenizers(
            src_texts=train_texts_src,
            tgt_texts=train_texts_tgt,
            output_dir=str(tokenizer_dir),
            src_vocab_size=self.config.src_vocab_size,
            tgt_vocab_size=self.config.tgt_vocab_size,
            model_type="bpe"
        )
        
        self.logger.log(f"Source vocab size: {self.tokenizer_src.vocab_size}")
        self.logger.log(f"Target vocab size: {self.tokenizer_tgt.vocab_size}")
    
    def setup_model(self):
        """Initialize model."""
        self.logger.log("Initializing model...")
        
        self.model = Transformer(
            src_vocab_size=self.tokenizer_src.vocab_size,
            tgt_vocab_size=self.tokenizer_tgt.vocab_size,
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            num_encoder_layers=self.config.num_encoder_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            d_ff=self.config.d_ff,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout,
            src_padding_idx=SentencePieceTokenizer.PAD_IDX,
            tgt_padding_idx=SentencePieceTokenizer.PAD_IDX
        )
        self.model = self.model.to(self.device)
        
        self.logger.log(f"Model parameters: {self.model.count_parameters():,}")
    
    def setup_optimizer(self):
        """Initialize optimizer and scheduler."""
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Warmup scheduler
        def lr_lambda(step):
            warmup = self.config.warmup_steps
            if step == 0:
                return 1e-8
            return min(step ** (-0.5), step * warmup ** (-1.5))
        
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
    
    def setup_criterion(self):
        """Initialize loss function."""
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=SentencePieceTokenizer.PAD_IDX,
            label_smoothing=self.config.label_smoothing
        )
    
    def setup(self):
        """Full setup: load data, build tokenizers, initialize model."""
        self.logger.log("="*60)
        self.logger.log("Setting up training...")
        self.logger.log(f"Data source: {self.config.data_source}")
        self.logger.log("="*60)
        
        # Load training data based on source type
        if self.config.data_source == "processed":
            # Load pre-processed data (fastest)
            self.logger.log("Loading pre-processed data...")
            
            # Helper to convert config path to actual path
            def get_data_path(config_path: str):
                """Convert config path (data/...) to actual path using data_dir."""
                if config_path.startswith('data/'):
                    relative_path = config_path[5:]  # Remove 'data/'
                else:
                    relative_path = config_path
                return self.config.paths.data_dir / relative_path
            
            train_path = get_data_path(self.config.processed_train)
            val_path = get_data_path(self.config.processed_val)
            
            train_dataset = ProcessedDataset(str(train_path))
            val_dataset = ProcessedDataset(str(val_path))
            
            # Load pre-trained tokenizers
            self.load_tokenizers()
            
        elif self.config.data_source == "local":
            self.logger.log("Loading from local files...")
            
            # Helper to convert config path to actual path
            def get_data_path(config_path: str):
                """Convert config path (data/...) to actual path using data_dir."""
                if config_path.startswith('data/'):
                    relative_path = config_path[5:]  # Remove 'data/'
                else:
                    relative_path = config_path
                return self.config.paths.data_dir / relative_path
            
            # Get paths using data_dir
            train_src = get_data_path(self.config.train_src)
            train_tgt = get_data_path(self.config.train_tgt)
            val_src = get_data_path(self.config.val_src)
            val_tgt = get_data_path(self.config.val_tgt)
            
            # Load datasets
            train_dataset = LocalTranslationDataset(
                src_file=str(train_src),
                tgt_file=str(train_tgt),
                max_samples=self.config.max_samples
            )
            val_dataset = LocalTranslationDataset(
                src_file=str(val_src),
                tgt_file=str(val_tgt)
            )
            
            # Get texts for tokenizer training
            src_texts, tgt_texts = train_dataset.get_all_texts()
            
            # Build tokenizers
            self.setup_tokenizers(src_texts, tgt_texts)
            
            # Update datasets with tokenizers
            train_dataset.tokenizer_src = self.tokenizer_src
            train_dataset.tokenizer_tgt = self.tokenizer_tgt
            val_dataset.tokenizer_src = self.tokenizer_src
            val_dataset.tokenizer_tgt = self.tokenizer_tgt
            
        else:
            # Load from HuggingFace
            self.logger.log("Loading from Hugging Face...")
            train_dataset = TranslationDataset(
                dataset_name=self.config.dataset_name,
                split="train",
                token=self.config.hf_token,
                max_samples=self.config.max_samples
            )
            val_dataset = TranslationDataset(
                dataset_name=self.config.dataset_name,
                split="validation",
                token=self.config.hf_token
            )
            
            # Extract texts for tokenizer training
            src_texts = [train_dataset[i]['src_text'] for i in range(len(train_dataset))]
            tgt_texts = [train_dataset[i]['tgt_text'] for i in range(len(train_dataset))]
            
            # Build tokenizers
            self.setup_tokenizers(src_texts, tgt_texts)
            
            # Update datasets with tokenizers
            train_dataset.tokenizer_src = self.tokenizer_src
            train_dataset.tokenizer_tgt = self.tokenizer_tgt
            val_dataset.tokenizer_src = self.tokenizer_src
            val_dataset.tokenizer_tgt = self.tokenizer_tgt
        
        # Create DataLoaders
        collate_fn = create_collate_fn(
            src_pad_idx=SentencePieceTokenizer.PAD_IDX,
            tgt_pad_idx=SentencePieceTokenizer.PAD_IDX
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        self.logger.log(f"Train batches: {len(self.train_loader)}")
        self.logger.log(f"Val batches: {len(self.val_loader)}")
        
        # Setup model, optimizer, criterion
        self.setup_model()
        self.setup_optimizer()
        self.setup_criterion()
        
        self.logger.log("✓ Setup complete!")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        metrics = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        # Zero gradients at start of epoch
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            # Teacher forcing: input is tgt[:-1], target is tgt[1:]
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]
            
            # Forward
            output = self.model(src, tgt_input)
            
            # Compute loss (scaled for gradient accumulation)
            output = output.reshape(-1, output.size(-1))
            tgt_target = tgt_target.reshape(-1)
            loss = self.criterion(output, tgt_target)
            loss = loss / self.config.gradient_accumulation_steps  # Scale loss
            
            # Backward (accumulate gradients)
            loss.backward()
            
            # Update weights every N steps
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update metrics (use unscaled loss for logging)
            unscaled_loss = loss.item() * self.config.gradient_accumulation_steps
            num_tokens = (tgt_target != SentencePieceTokenizer.PAD_IDX).sum().item()
            metrics.update(unscaled_loss, num_tokens)
            
            # Update progress bar with more info
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f"{metrics.avg_loss:.4f}",
                'ppl': f"{metrics.perplexity:.1f}",
                'lr': f"{current_lr:.2e}"
            })
            
            self.global_step += 1
        
        return {'loss': metrics.avg_loss, 'perplexity': metrics.perplexity}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        metrics = MetricsTracker()
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]
            
            output = self.model(src, tgt_input)
            
            output = output.reshape(-1, output.size(-1))
            tgt_target = tgt_target.reshape(-1)
            loss = self.criterion(output, tgt_target)
            
            num_tokens = (tgt_target != SentencePieceTokenizer.PAD_IDX).sum().item()
            metrics.update(loss.item(), num_tokens)
        
        return {'loss': metrics.avg_loss, 'perplexity': metrics.perplexity}
    
    def train(self, resume_from: Optional[str] = None):
        """
        Full training loop.
        
        Args:
            resume_from: Optional checkpoint path to resume from
        """
        # Resume if specified
        if resume_from:
            checkpoint = load_checkpoint(
                resume_from, self.model, self.optimizer, self.device
            )
            self.current_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.log(f"Starting training from epoch {self.current_epoch}")
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log
            self.logger.log_epoch(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                val_metrics['loss'],
                str(self.config.paths.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"),
                best_val_loss=self.best_val_loss
            )
            
            if is_best:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_metrics['loss'],
                    str(self.config.paths.checkpoint_dir / "best_model.pt"),
                    best_val_loss=self.best_val_loss
                )
                self.logger.log("✓ Saved new best model!")
        
        self.logger.log("Training complete!")
        self.logger.close()


def main(config_path: Optional[str] = None, **overrides):
    """Main training function."""
    trainer = Trainer(config_path=config_path, overrides=overrides)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
