"""
Logging utilities for training.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def setup_logger(
    name: str = "translation",
    log_dir: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with console and optional file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files (optional)
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"train_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to: {log_file}")
    
    return logger


class TrainingLogger:
    """
    Training logger with TensorBoard support.
    """
    
    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        logger_name: str = "training"
    ):
        """
        Args:
            log_dir: Directory for logs
            use_tensorboard: Whether to use TensorBoard
            logger_name: Name for the logger
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup text logger
        self.logger = setup_logger(logger_name, log_dir)
        
        # Setup TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
                self.logger.info("TensorBoard enabled")
            except ImportError:
                self.logger.warning("TensorBoard not available")
        
        self.step = 0
    
    def log(self, message: str, level: str = "info"):
        """Log a message."""
        getattr(self.logger, level)(message)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """
        Log metrics to console and TensorBoard.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Global step (uses internal counter if None)
            prefix: Prefix for metric names
        """
        if step is None:
            step = self.step
            self.step += 1
        
        # Log to console
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} | {metric_str}")
        
        # Log to TensorBoard
        if self.writer is not None:
            for name, value in metrics.items():
                tag = f"{prefix}/{name}" if prefix else name
                self.writer.add_scalar(tag, value, step)
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Log epoch summary."""
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Epoch {epoch} Summary")
        
        # Train metrics
        train_str = " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        self.logger.info(f"  Train: {train_str}")
        
        # Val metrics
        if val_metrics:
            val_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            self.logger.info(f"  Val:   {val_str}")
        
        self.logger.info(f"{'='*60}")
        
        # TensorBoard
        if self.writer is not None:
            for name, value in train_metrics.items():
                self.writer.add_scalar(f"train/{name}", value, epoch)
            if val_metrics:
                for name, value in val_metrics.items():
                    self.writer.add_scalar(f"val/{name}", value, epoch)
    
    def close(self):
        """Close logger and TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
