"""Reusable evaluation helpers for notebooks.

This module exposes functions that the notebook `notebooks/05_evaluation.ipynb`
can import and call. It does NOT run anything on import.
"""
import json
from pathlib import Path
import pickle
import time
from typing import Optional, Dict, Any, List, Tuple

import sentencepiece as spm
import torch
from sacrebleu import corpus_bleu

from src.model import Transformer
from src.utils import get_device


def load_tokenizers_and_config(project_root: Optional[Path] = None) -> Tuple[Dict[str, Any], spm.SentencePieceProcessor, spm.SentencePieceProcessor]:
    """Load `tokenizer_info.json` and SentencePiece processors.

    Args:
        project_root: path to project root; if None auto-detected.
    Returns:
        tokenizer_info, sp_vi, sp_en
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    tokenizer_info_path = project_root / 'data' / 'processed' / 'tokenizer_info.json'
    with open(tokenizer_info_path, 'r', encoding='utf-8') as f:
        tokenizer_info = json.load(f)

    # Resolve sentencepiece model paths robustly. tokenizer_info may contain
    # paths that are relative to different working directories (notebook vs project).
    def _resolve_model_path(key: str) -> Path:
        raw = Path(tokenizer_info[key])
        candidates = [raw]
        # candidate relative to project root
        candidates.append(project_root / raw)
        # common location: project_root/data/processed/<name>
        candidates.append(project_root / 'data' / 'processed' / raw.name)
        for p in candidates:
            try:
                if p.exists():
                    return p
            except Exception:
                # ignore permission / unusual path errors and continue
                pass
        # If none found, return the last candidate (will raise when loading)
        return candidates[-1]

    vi_path = _resolve_model_path('vi_model')
    en_path = _resolve_model_path('en_model')

    sp_vi = spm.SentencePieceProcessor()
    sp_en = spm.SentencePieceProcessor()
    if not vi_path.exists():
        tried = [str(p) for p in [Path(tokenizer_info['vi_model']), project_root / Path(tokenizer_info['vi_model']), project_root / 'data' / 'processed' / Path(tokenizer_info['vi_model']).name]]
        raise OSError(f"spm_vi.model not found. Tried: {tried}")
    if not en_path.exists():
        tried = [str(p) for p in [Path(tokenizer_info['en_model']), project_root / Path(tokenizer_info['en_model']), project_root / 'data' / 'processed' / Path(tokenizer_info['en_model']).name]]
        raise OSError(f"spm_en.model not found. Tried: {tried}")

    sp_vi.load(str(vi_path))
    sp_en.load(str(en_path))

    return tokenizer_info, sp_vi, sp_en


def build_and_load_model(checkpoint_path: str, tokenizer_info: Dict[str, Any], device: Optional[torch.device] = None, model_kwargs: Optional[Dict[str, Any]] = None) -> Transformer:
    """Instantiate the `Transformer` and load the checkpoint.

    Args:
        checkpoint_path: path to checkpoint file (can be relative to project root)
        tokenizer_info: dict loaded from `tokenizer_info.json`
        device: torch device; if None uses `get_device()`
        model_kwargs: override hyperparameters for model construction
    Returns:
        model (on device)
    """
    project_root = Path(__file__).resolve().parents[1]
    if device is None:
        device = get_device()

    defaults = dict(
        src_vocab_size=tokenizer_info['vi_vocab_size'],
        tgt_vocab_size=tokenizer_info['en_vocab_size'],
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_len=tokenizer_info['max_length'],
        dropout=0.1,
        pad_idx=tokenizer_info['pad_id'],
    )
    if model_kwargs:
        defaults.update(model_kwargs)

    model = Transformer(**defaults)

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        ckpt = project_root / checkpoint_path
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(str(ckpt), map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def _decode_with_sentencepiece(sp: spm.SentencePieceProcessor, ids: List[int]) -> str:
    """Robust decode handling different sentencepiece API names."""
    try:
        return sp.decode_ids(ids)
    except Exception:
        try:
            return sp.DecodeIds(ids)
        except Exception:
            if hasattr(sp, 'IdToPiece'):
                pieces = [sp.IdToPiece(int(i)) for i in ids]
            elif hasattr(sp, 'id_to_piece'):
                pieces = [sp.id_to_piece(int(i)) for i in ids]
            else:
                pieces = []
            return ''.join(pieces).replace('▁', ' ').strip()


def greedy_decode(model: Transformer, src_ids: List[int], sp_en: spm.SentencePieceProcessor, tokenizer_info: Dict[str, Any], device: torch.device, max_len: Optional[int] = None) -> str:
    """Greedy decode a single source sentence (ids include BOS/EOS expectations).

    Returns the decoded string (without BOS/EOS).
    """
    if max_len is None:
        max_len = tokenizer_info.get('max_length', 128)

    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        encoder_output, src_mask = model.encode(src)

    bos = tokenizer_info['bos_id']
    eos = tokenizer_info['eos_id']

    tgt_ids = [bos]
    for _ in range(max_len):
        tgt = torch.tensor(tgt_ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model.decode(tgt, encoder_output, src_mask)
        logits = out[0, -1, :]
        next_id = int(torch.argmax(logits).item())
        tgt_ids.append(next_id)
        if next_id == eos:
            break

    # remove BOS and EOS
    if tgt_ids and tgt_ids[0] == bos:
        decoded_ids = tgt_ids[1:]
    else:
        decoded_ids = tgt_ids

    if decoded_ids and decoded_ids[-1] == eos:
        decoded_ids = decoded_ids[:-1]

    return _decode_with_sentencepiece(sp_en, decoded_ids)


def load_test_split(project_root: Optional[Path] = None):
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    splits_path = project_root / 'data' / 'processed' / 'splits.pkl'
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    return splits['test']


def evaluate_checkpoint(checkpoint_path: str, project_root: Optional[Path] = None, device: Optional[torch.device] = None, model_kwargs: Optional[Dict[str, Any]] = None, max_examples: Optional[int] = None) -> Dict[str, Any]:
    """Evaluate a checkpoint on the test split and return BLEU and outputs.

    Returns dict: { 'bleu': float, 'hyps': list, 'refs': list }
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    if device is None:
        device = get_device()

    tokenizer_info, sp_vi, sp_en = load_tokenizers_and_config(project_root)
    model = build_and_load_model(checkpoint_path, tokenizer_info, device=device, model_kwargs=model_kwargs)

    test_data = load_test_split(project_root)
    hyps = []
    refs = []

    start = time.time()
    for i, item in enumerate(test_data):
        if max_examples is not None and i >= max_examples:
            break

        src_text = item.get('vi') if 'vi' in item else item.get('src', '')
        ref_text = item.get('en') if 'en' in item else item.get('tgt', '')

        src_ids = sp_vi.encode_as_ids(src_text)
        src_ids = [tokenizer_info['bos_id']] + src_ids + [tokenizer_info['eos_id']]
        if len(src_ids) > tokenizer_info['max_length']:
            src_ids = src_ids[:tokenizer_info['max_length']-1] + [tokenizer_info['eos_id']]

        hyp = greedy_decode(model, src_ids, sp_en, tokenizer_info, device)
        hyps.append(hyp)
        refs.append(ref_text)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"Decoded {i+1}/{len(test_data)} in {elapsed:.1f}s")

    bleu = corpus_bleu(hyps, [refs])
    return {'bleu': bleu.score, 'hyps': hyps, 'refs': refs}

