"""
Evaluation metrics for translation.
"""

from typing import List, Optional
import sacrebleu


def compute_bleu(
    predictions: List[str],
    references: List[str],
    lowercase: bool = False
) -> dict:
    """
    Compute BLEU score using SacreBLEU.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        lowercase: Whether to lowercase before computing
    
    Returns:
        Dictionary with BLEU score and details
    """
    # SacreBLEU expects references as list of lists
    refs = [[ref] for ref in references]
    
    # Compute BLEU
    bleu = sacrebleu.corpus_bleu(
        predictions,
        refs,
        lowercase=lowercase
    )
    
    return {
        'bleu': bleu.score,
        'precisions': bleu.precisions,
        'bp': bleu.bp,  # Brevity penalty
        'ratio': bleu.sys_len / bleu.ref_len if bleu.ref_len > 0 else 0,
        'sys_len': bleu.sys_len,
        'ref_len': bleu.ref_len,
    }


def compute_sentence_bleu(
    prediction: str,
    reference: str,
    lowercase: bool = False
) -> float:
    """
    Compute sentence-level BLEU score.
    
    Args:
        prediction: Predicted translation
        reference: Reference translation
        lowercase: Whether to lowercase
    
    Returns:
        BLEU score
    """
    bleu = sacrebleu.sentence_bleu(
        prediction,
        [reference],
        lowercase=lowercase
    )
    return bleu.score


class MetricsTracker:
    """Track and aggregate metrics during training/evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_loss = 0.0
        self.total_tokens = 0
        self.num_batches = 0
        self.predictions = []
        self.references = []
    
    def update(
        self,
        loss: float,
        num_tokens: int,
        predictions: Optional[List[str]] = None,
        references: Optional[List[str]] = None
    ):
        """Update metrics with batch results."""
        self.total_loss += loss
        self.total_tokens += num_tokens
        self.num_batches += 1
        
        if predictions is not None:
            self.predictions.extend(predictions)
        if references is not None:
            self.references.extend(references)
    
    @property
    def avg_loss(self) -> float:
        """Get average loss."""
        if self.num_batches == 0:
            return 0.0
        return self.total_loss / self.num_batches
    
    @property
    def perplexity(self) -> float:
        """Get perplexity."""
        import math
        if self.total_tokens == 0:
            return float('inf')
        return math.exp(self.total_loss / self.total_tokens)
    
    def compute_bleu(self) -> float:
        """Compute BLEU score from accumulated predictions."""
        if not self.predictions or not self.references:
            return 0.0
        return compute_bleu(self.predictions, self.references)['bleu']
    
    def get_summary(self) -> dict:
        """Get summary of all metrics."""
        summary = {
            'loss': self.avg_loss,
            'perplexity': self.perplexity,
            'num_batches': self.num_batches,
        }
        
        if self.predictions and self.references:
            summary['bleu'] = self.compute_bleu()
        
        return summary
