# Translate Transformer NLP Project

Dự án xây dựng mô hình dịch máy Seq2Seq với kiến trúc Transformer từ đầu (from scratch) cho bài tập lớn.

## Cấu trúc Project

```
Translate_Transformer_NLP_Project/
├── data/                          # Dữ liệu
│   ├── raw/                       # Data gốc (cache từ Hugging Face)
│   └── processed/                 # Vocab, tokenized data
├── checkpoints/                   # Model checkpoints
├── logs/                          # Training logs, tensorboard
├── results/                       # Evaluation results, plots
├── src/                           # Source code modules
│   ├── attention.py               # Attention mechanisms
│   ├── layers.py                  # Encoder/Decoder layers
│   ├── model.py                   # Full Transformer model
│   └── utils.py                   # Helper functions
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # Phân tích & EDA
│   ├── 02_preprocessing.ipynb     # Tiền xử lý data
│   ├── 03_model_building.ipynb    # Build & test model
│   ├── 04_training.ipynb          # Training loop
│   └── 05_evaluation.ipynb        # Evaluation & visualization
└── requirements.txt               # Dependencies
```

## Dataset

- **PHOMT** (Vietnamese-English): Dataset song ngữ Việt-Anh từ VietAI
- Subset: 300K-500K câu
- Max sequence length: 128 tokens
- Load trực tiếp từ Hugging Face Datasets

## Model Configuration

```python
{
    'd_model': 512,
    'num_heads': 8,
    'num_encoder_layers': 4-6,
    'num_decoder_layers': 4-6,
    'd_ff': 2048,
    'max_len': 128,
    'dropout': 0.1,
    'batch_size': 8,
    'gradient_accumulation_steps': 4
}
```

## Hardware Requirements

- GPU: RTX 4060 Laptop (8GB VRAM)
- RAM: 16GB
- Training time: ~10-25 giờ (5-10 epochs)

## Installation

```bash
pip install -r requirements.txt
```

## Các Thành Phần Chính

### 1. Data Processing (notebooks/01-02)
- Thu thập và làm sạch dữ liệu PHOMT
- Tokenization (BPE hoặc word-level)
- Xây dựng vocabulary
- Padding/Truncation
- DataLoader

### 2. Model Architecture (src/ & notebooks/03)
- **Scaled Dot-Product Attention**
- **Multi-Head Attention**
- **Positional Encoding** (Sinusoidal)
- **Encoder Layer**: Self-Attention + FFN
- **Decoder Layer**: Masked Self-Attention + Cross-Attention + FFN
- **Full Transformer Model**

### 3. Training (notebooks/04)
- Loss function: Cross-Entropy với Label Smoothing
- Optimizer: Adam/AdamW
- Learning Rate Scheduler: Warmup + Decay
- Mixed Precision Training (FP16)
- Gradient Accumulation
- Checkpointing

### 4. Evaluation (notebooks/05)
- Decoding strategies:
  - Greedy Search
  - Beam Search
- Metrics:
  - BLEU Score
  - Perplexity
- Attention visualization
- Error analysis

## Usage

### Bước 1: Khám phá dữ liệu
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Bước 2: Tiền xử lý
```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

### Bước 3: Xây dựng model
```bash
jupyter notebook notebooks/03_model_building.ipynb
```

### Bước 4: Training
```bash
jupyter notebook notebooks/04_training.ipynb
```

### Bước 5: Evaluation
```bash
jupyter notebook notebooks/05_evaluation.ipynb
```

## Features

- ✅ Transformer architecture from scratch
- ✅ Scaled Dot-Product Attention
- ✅ Multi-Head Attention
- ✅ Positional Encoding
- ✅ Encoder-Decoder architecture
- ✅ Beam Search decoding
- ✅ BLEU score evaluation
- ✅ Mixed precision training
- ✅ Gradient accumulation
- ✅ Learning rate warmup
- ✅ Label smoothing

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [PHOMT Dataset](https://huggingface.co/datasets/VietAI/phomt) - VietAI Vietnamese-English dataset
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

## License

MIT License

## Author

Bài tập lớn - Xử lý Ngôn ngữ Tự nhiên
