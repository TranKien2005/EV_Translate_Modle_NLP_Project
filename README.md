# 🌐 Vietnamese-English Translation Model

Transformer-based Neural Machine Translation for Vietnamese ↔ English.

## ✨ Features

- **VI → EN**: Vietnamese to English translation
- **EN → VI**: English to Vietnamese translation
- **Transformer architecture**: 6 encoder + 5 decoder layers
- **SentencePiece BPE tokenizer**: 16k vocabulary

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/TranKien2005/EV_Translate_Modle_NLP_Project.git
cd EV_Translate_Modle_NLP_Project
pip install torch sentencepiece sacrebleu google-generativeai python-dotenv tqdm pyyaml tensorboard
```

### API Keys (Required for full features)

Create `.env` file in project root:
```bash
GEMINI_API_KEY=your_gemini_api_key    # For Gemini evaluation
HF_TOKEN=your_huggingface_token       # For dataset download
```

- **GEMINI_API_KEY**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **HF_TOKEN**: Get from [HuggingFace Settings](https://huggingface.co/settings/tokens)

### Translation

```python
from src.evaluate import load_translator

# VI → EN
translator = load_translator(
    checkpoint_path='checkpoints_vi_en/best_model.pt',
    vocab_src_path='checkpoints_vi_en/tokenizers/tokenizer_vi.model',
    vocab_tgt_path='checkpoints_vi_en/tokenizers/tokenizer_en.model',
    config_path='config/config_vi_en.yaml'
)

result = translator.translate("Xin chào, bạn khỏe không?")
print(result)  # Hello, how are you?
```

## 🏋️ Training

### On Kaggle (Recommended)
1. Upload notebook `notebooks/train_kaggle_vi_en.ipynb` to Kaggle
2. Enable GPU accelerator
3. Run all cells

### Resume Training
1. Upload checkpoint + tokenizers + processed data
2. Use `notebooks/train_kaggle_vi_en_resume.ipynb`

### Local Training

**1. Download dataset:**
```bash
python scripts/download_phomt.py
```

**2. Preprocess data:**
```bash
# For VI → EN
python scripts/preprocess_data.py --config config/config_vi_en.yaml

# For EN → VI
python scripts/preprocess_data.py --config config/config.yaml
```

**3. Train:**
```bash
python -m src.train --config config/config_vi_en.yaml
```

## ⚙️ Configuration

### Config Files

| File | Direction | Description |
|------|-----------|-------------|
| `config/config.yaml` | EN → VI | English to Vietnamese |
| `config/config_vi_en.yaml` | VI → EN | Vietnamese to English |

### Key Parameters

```yaml
# Model
model:
  d_model: 512              # Model dimension
  num_heads: 8              # Attention heads
  num_encoder_layers: 6     # Encoder depth
  num_decoder_layers: 5     # Decoder depth

# Training
training:
  batch_size: 64            # Batch size per GPU
  gradient_accumulation_steps: 2  # Effective batch = 128
  learning_rate: 0.0005     # Initial learning rate
  epochs: 15                # Training epochs
  warmup_steps: 2000        # LR warmup steps

# Data
data:
  source: "local"           # "local", "huggingface", or "processed"
  max_seq_len: 128          # Maximum sequence length
```

## 🔧 Scripts

| Script | Description |
|--------|-------------|
| `scripts/download_phomt.py` | Download PhoMT dataset from HuggingFace |
| `scripts/preprocess_data.py` | Tokenize and prepare training data |

### Preprocess Options
```bash
python scripts/preprocess_data.py --help

# Options:
#   --config PATH       Config file (default: config/config.yaml)
#   --max-samples N     Limit samples for testing
#   --force-retrain     Retrain tokenizers even if exists
```

## � Evaluation

### BLEU Score
```bash
python -m src.evaluate \
    --config config/config_vi_en.yaml \
    --checkpoint checkpoints_vi_en/best_model.pt
```

### Gemini Evaluation (requires GEMINI_API_KEY)
```bash
python -m src.evaluate \
    --config config/config_vi_en.yaml \
    --checkpoint checkpoints_vi_en/best_model.pt \
    --gemini
```

### Interactive Translation
```bash
python -m src.evaluate \
    --config config/config_vi_en.yaml \
    --checkpoint checkpoints_vi_en/best_model.pt \
    --interactive
```

## �📁 Project Structure

```
├── config/          # Training configurations
├── notebooks/       # Kaggle training notebooks
├── src/
│   ├── models/      # Transformer model
│   ├── data/        # Dataset & tokenizer
│   └── train.py     # Training script
└── scripts/         # Utility scripts
```

## 📄 License

MIT License
