"""
Evaluation and inference script for Transformer translation model.
"""

import torch
import json
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
from tqdm import tqdm

from src.config import Config, load_config
from src.models import Transformer
from src.data.tokenizer import SentencePieceTokenizer
from src.utils import get_device
from src.utils.metrics import compute_bleu


class Translator:
    """
    Translator class for inference.
    """
    
    def __init__(
        self,
        model: Transformer,
        tokenizer_src: SentencePieceTokenizer,
        tokenizer_tgt: SentencePieceTokenizer,
        device: str = "cpu",
        max_len: int = 128
    ):
        """
        Args:
            model: Trained Transformer model
            tokenizer_src: Source tokenizer
            tokenizer_tgt: Target tokenizer
            device: Device to use
            max_len: Maximum generation length
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.device = device
        self.max_len = max_len
    
    @torch.no_grad()
    def translate(
        self,
        text: str,
        beam_size: int = 4,
        return_attention: bool = False
    ) -> str:
        """
        Translate a single sentence using beam search.
        
        Args:
            text: Source text to translate
            beam_size: Beam size (1 = greedy, >1 = beam search)
            return_attention: Whether to return attention weights
        
        Returns:
            Translated text
        """
        # Tokenize source
        src_tokens = self.tokenizer_src.encode(text)
        src = torch.tensor([src_tokens]).to(self.device)
        
        # Encode source
        src_mask = self.model.create_src_mask(src)
        encoder_output = self.model.encode(src, src_mask)
        
        if beam_size == 1:
            # Greedy decoding (faster)
            return self._greedy_decode(encoder_output, src_mask)
        else:
            # Beam search (better quality)
            return self._beam_search(encoder_output, src_mask, beam_size)
    
    def _greedy_decode(self, encoder_output, src_mask) -> str:
        """Greedy decoding - fast but lower quality."""
        tgt_tokens = [SentencePieceTokenizer.BOS_IDX]
        
        for _ in range(self.max_len):
            tgt = torch.tensor([tgt_tokens]).to(self.device)
            tgt_mask = self.model.create_tgt_mask(tgt)
            
            output = self.model.decode(tgt, encoder_output, src_mask, tgt_mask)
            next_token = output[0, -1].argmax().item()
            tgt_tokens.append(next_token)
            
            if next_token == SentencePieceTokenizer.EOS_IDX:
                break
        
        return self.tokenizer_tgt.decode(tgt_tokens)
    
    def _beam_search(self, encoder_output, src_mask, beam_size: int = 4) -> str:
        """Beam search decoding - slower but better quality."""
        import torch.nn.functional as F
        
        # Initialize beams: list of (tokens, log_prob)
        beams = [([SentencePieceTokenizer.BOS_IDX], 0.0)]
        
        # Expand encoder output for beam search
        encoder_output = encoder_output.expand(beam_size, -1, -1)
        src_mask = src_mask.expand(beam_size, -1, -1, -1) if src_mask is not None else None
        
        for step in range(self.max_len):
            all_candidates = []
            
            for tokens, score in beams:
                # Skip completed sequences
                if tokens[-1] == SentencePieceTokenizer.EOS_IDX:
                    all_candidates.append((tokens, score))
                    continue
                
                # Decode
                tgt = torch.tensor([tokens]).to(self.device)
                tgt_mask = self.model.create_tgt_mask(tgt)
                
                output = self.model.decode(
                    tgt, encoder_output[:1], 
                    src_mask[:1] if src_mask is not None else None, 
                    tgt_mask
                )
                
                # Get log probabilities for next token
                log_probs = F.log_softmax(output[0, -1], dim=-1)
                
                # Get top-k candidates
                topk_log_probs, topk_indices = log_probs.topk(beam_size)
                
                for log_prob, idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                    new_tokens = tokens + [idx]
                    new_score = score + log_prob
                    all_candidates.append((new_tokens, new_score))
            
            # Select top beam_size candidates
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_size]
            
            # Check if all beams are complete
            if all(b[0][-1] == SentencePieceTokenizer.EOS_IDX for b in beams):
                break
        
        # Return best sequence
        best_tokens = beams[0][0]
        return self.tokenizer_tgt.decode(best_tokens)
    
    @torch.no_grad()
    def translate_batch(self, texts: List[str], beam_size: int = 4) -> List[str]:
        """Translate a batch of sentences."""
        return [self.translate(text, beam_size=beam_size) for text in texts]


def load_translator(
    checkpoint_path: str,
    vocab_src_path: str,
    vocab_tgt_path: str,
    config_path: Optional[str] = None,
    device: str = "auto"
) -> Translator:
    """
    Load a trained translator.
    
    Args:
        checkpoint_path: Path to model checkpoint
        vocab_src_path: Path to source vocabulary JSON
        vocab_tgt_path: Path to target vocabulary JSON
        config_path: Optional path to config file
        device: Device to use
    
    Returns:
        Configured Translator instance
    """
    # Load config
    config = load_config(config_path) if config_path else None
    device = get_device(device)
    
    # Load tokenizers (SentencePiece models)
    tokenizer_src = SentencePieceTokenizer(vocab_src_path)
    tokenizer_tgt = SentencePieceTokenizer(vocab_tgt_path)
    
    # Create model
    model = Transformer(
        src_vocab_size=tokenizer_src.vocab_size,
        tgt_vocab_size=tokenizer_tgt.vocab_size,
        d_model=config.d_model if config else 512,
        num_heads=config.num_heads if config else 8,
        num_encoder_layers=config.num_encoder_layers if config else 6,
        num_decoder_layers=config.num_decoder_layers if config else 6,
        d_ff=config.d_ff if config else 2048,
        max_seq_len=config.max_seq_len if config else 128,
        dropout=0.0,  # No dropout at inference
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from {checkpoint_path}")
    
    return Translator(
        model=model,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        device=str(device),
        max_len=config.max_seq_len if config else 128
    )


def save_evaluation_results(
    results: Dict,
    predictions: List[str],
    source_texts: List[str],
    reference_texts: List[str],
    output_dir: str,
    checkpoint_name: str = "model",
    num_examples: int = 10
):
    """
    Save evaluation results to files with sample examples.
    
    Args:
        results: Evaluation results (BLEU, Gemini, etc.)
        predictions: List of model predictions
        source_texts: List of source sentences
        reference_texts: List of reference translations
        output_dir: Output directory
        checkpoint_name: Name of checkpoint for file naming
        num_examples: Number of examples to display (default: 10)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save metrics to JSON
    metrics = {
        "bleu": results.get("bleu", 0),
        "precisions": results.get("precisions", []),
        "brevity_penalty": results.get("bp", 0),
        "gemini_score": results.get("gemini_score", None),
        "num_samples": len(predictions),
        "timestamp": timestamp,
        "checkpoint": checkpoint_name
    }
    
    metrics_path = output_dir / f"eval_{timestamp}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✓ Metrics saved to: {metrics_path}")
    
    # Save all translations
    translations_dir = output_dir / "translations"
    translations_dir.mkdir(exist_ok=True)
    
    translations_path = translations_dir / f"translations_{timestamp}.txt"
    with open(translations_path, 'w', encoding='utf-8') as f:
        for i, (src, ref, pred) in enumerate(zip(source_texts, reference_texts, predictions)):
            f.write(f"[{i+1}]\n")
            f.write(f"SRC: {src}\n")
            f.write(f"REF: {ref}\n")
            f.write(f"PRD: {pred}\n")
            f.write("-" * 50 + "\n")
    print(f"✓ Translations saved to: {translations_path}")
    
    # Print sample examples to console
    print(f"\n{'='*60}")
    print(f"📝 Sample Translations ({min(num_examples, len(predictions))} examples)")
    print(f"{'='*60}")
    
    for i in range(min(num_examples, len(predictions))):
        print(f"\n[Example {i+1}]")
        print(f"  🔹 SRC: {source_texts[i][:80]}{'...' if len(source_texts[i]) > 80 else ''}")
        print(f"  🔹 REF: {reference_texts[i][:80]}{'...' if len(reference_texts[i]) > 80 else ''}")
        print(f"  🔹 PRD: {predictions[i][:80]}{'...' if len(predictions[i]) > 80 else ''}")
    
    print(f"\n{'='*60}")
    
    return metrics_path, translations_path


from src.utils.gemini_eval import GeminiEvaluator

def evaluate_model(
    translator: Translator,
    source_texts: List[str],
    reference_texts: List[str],
    config: Optional[Config] = None,
    output_dir: Optional[str] = None,
    checkpoint_name: str = "model",
    run_gemini: bool = False
) -> Dict[str, float]:
    """
    Evaluate model on a test set using BLEU and optionally Gemini.
    
    Args:
        translator: Translator instance
        source_texts: List of source sentences
        reference_texts: List of reference translations
        config: Config object for limits
        output_dir: Optional directory to save results
        checkpoint_name: Name of checkpoint for naming files
        run_gemini: Whether to run Gemini evaluation
    
    Returns:
        Dictionary with scores
    """
    # 1. BLEU Evaluation
    # Limit samples for BLEU if configured
    max_bleu = config.eval_max_samples_bleu if config else 1000
    n_bleu = min(len(source_texts), max_bleu)
    
    print(f"\nEvaluating BLEU on first {n_bleu} samples...")
    predictions = []
    
    # Translate batch for BLEU
    # Using a subset for speed
    subset_src = source_texts[:n_bleu]
    subset_ref = reference_texts[:n_bleu]
    
    for text in tqdm(subset_src):
        pred = translator.translate(text)
        predictions.append(pred)
    
    # Compute BLEU
    bleu_result = compute_bleu(predictions, subset_ref)
    
    print(f"\n{'='*40}")
    print(f"BLEU Score: {bleu_result['bleu']:.2f}")
    print(f"Precisions: {[f'{p:.1f}' for p in bleu_result['precisions']]}")
    print(f"Brevity Penalty: {bleu_result['bp']:.4f}")
    
    results = bleu_result
    
    # 2. Gemini Evaluation (Optional)
    if run_gemini:
        print(f"\nEvaluating Gemini Score...")
        try:
            gemini = GeminiEvaluator()
            max_gemini = config.eval_max_samples_gemini if config else 200
            
            # Use same predictions if n_bleu >= max_gemini, else translate more?
            # Typically max_gemini (200) < max_bleu (1000), so we reuse predictions
            
            gemini_src = subset_src[:max_gemini]
            gemini_ref = subset_ref[:max_gemini]
            gemini_pred = predictions[:max_gemini]
            
            gemini_results = gemini.evaluate_batch(
                sources=gemini_src,
                references=gemini_ref,
                candidates=gemini_pred,
                max_samples=max_gemini
            )
            
            results.update(gemini_results)
            print(f"Gemini Score: {gemini_results['gemini_score']:.2f} (on {gemini_results['num_samples']} samples)")
            
        except Exception as e:
            print(f"Gemini evaluation failed: {e}")
    
    print(f"{'='*40}")
    
    # Save results if output_dir is provided
    if output_dir:
        save_evaluation_results(
            results=results,
            predictions=predictions, # Saves all generated predictions (n_bleu)
            source_texts=subset_src,
            reference_texts=subset_ref,
            output_dir=output_dir,
            checkpoint_name=checkpoint_name
        )
    
    return results


def interactive_translate(translator: Translator):
    """
    Interactive translation mode.
    """
    print("\n" + "="*50)
    print("Interactive Translation Mode")
    print("Type 'quit' to exit")
    print("="*50 + "\n")
    
    while True:
        try:
            text = input("English: ").strip()
            if text.lower() == 'quit':
                break
            
            translation = translator.translate(text)
            print(f"Vietnamese: {translation}\n")
        
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate or translate")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--vocab-src", type=str, required=True, help="Source tokenizer .model path")
    parser.add_argument("--vocab-tgt", type=str, required=True, help="Target tokenizer .model path")
    parser.add_argument("--config", type=str, help="Config file path")
    
    # Modes
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--test", action="store_true", help="Evaluate on Test set")
    parser.add_argument("--val", action="store_true", help="Evaluate on Validation set")
    parser.add_argument("--gemini", action="store_true", help="Enable Gemini evaluation")
    
    parser.add_argument("--device", type=str, default="auto", help="Device")
    
    args = parser.parse_args()
    
    # Load config and translator
    config = load_config(args.config) if args.config else load_config()
    
    translator = load_translator(
        args.checkpoint,
        args.vocab_src,
        args.vocab_tgt,
        args.config,
        args.device
    )
    
    if args.interactive:
        interactive_translate(translator)
    
    elif args.test or args.val:
        # Helper to convert config path to actual path
        def get_data_path(config_path: str):
            """Convert config path (data/...) to actual path using data_dir."""
            if config_path.startswith('data/'):
                relative_path = config_path[5:]  # Remove 'data/'
            else:
                relative_path = config_path
            return config.paths.data_dir / relative_path
        
        if args.test:
            print("Loading TEST set...")
            if config.data_source == "processed":
                data_path = get_data_path(config.processed_test)
                ds = ProcessedDataset(str(data_path))
                src_texts = [translator.tokenizer_src.decode(ids) for ids in ds.src_tokens]
                tgt_texts = [translator.tokenizer_tgt.decode(ids) for ids in ds.tgt_tokens]
            else:
                src_path = get_data_path(config.test_src)
                tgt_path = get_data_path(config.test_tgt)
                src_texts, tgt_texts = LocalTranslationDataset.load_texts(str(src_path), str(tgt_path))
            
            output_dir = config.paths.log_dir / "test_results"
            
        else: # args.val
            print("Loading VALIDATION set...")
            if config.data_source == "processed":
                data_path = get_data_path(config.processed_val)
                ds = ProcessedDataset(str(data_path))
                src_texts = [translator.tokenizer_src.decode(ids) for ids in ds.src_tokens]
                tgt_texts = [translator.tokenizer_tgt.decode(ids) for ids in ds.tgt_tokens]
            else:
                src_path = get_data_path(config.val_src)
                tgt_path = get_data_path(config.val_tgt)
                src_texts, tgt_texts = LocalTranslationDataset.load_texts(str(src_path), str(tgt_path))
                
            output_dir = config.paths.log_dir / "val_results"

        # Run evaluation
        evaluate_model(
            translator=translator,
            source_texts=src_texts,
            reference_texts=tgt_texts,
            config=config,
            output_dir=str(output_dir),
            checkpoint_name=Path(args.checkpoint).stem,
            run_gemini=args.gemini
        )
        
    else:
        print("Please specify --interactive, --test, or --val")
