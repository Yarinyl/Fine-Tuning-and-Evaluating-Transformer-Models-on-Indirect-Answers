# Fine-Tuning and Evaluating Transformer Models on Indirect Answers and Sentiment Analysis Tasks

This repository contains a complete pipeline for training and evaluating transformer-based models (e.g., BERT, RoBERTa, and LLaMA 3) on several text classification tasks using Hugging Face and Unsloth.

## 📁 Project Structure

```
├── config_LLM.py        # Configuration and dataset loading for LLaMA models
├── config_SLM.py        # Configuration and dataset loading for BERT/RoBERTa models
├── train_test_LLM.py         # Training and evaluation logic for LLaMA models (Unsloth)
├── train_test_SLM.py         # Training and evaluation logic for BERT/RoBERTa models
```

## 🧠 Supported Models

- **LLaMA 3** (via [Unsloth](https://github.com/unslothai/unsloth))
- **BERT** (`bert-base-uncased`)
- **RoBERTa** (`roberta-base`)

## 📊 Supported Datasets

| Dataset  | Description                                       |
|----------|---------------------------------------------------|
| Circa    | Indirect answer classification (Yes/No)           |
| SST-2    | Sentiment classification (Positive/Negative)      |
| IMDB     | Movie review sentiment classification             |
| Amazon   | Product review sentiment classification           |

All datasets are loaded via the Hugging Face `datasets` library.

## 🔧 Configuration

Each training script loads its configuration from a dedicated file:

- `Config` class in `config_LLM.py` handles LLaMA-specific settings and Circa formatting.
- `Config` class in `config_SLM.py` handles BERT/RoBERTa model setup, tokenization, and dataset preparation.

The config includes:

- Model name and paradigm
- Experiment path
- Training arguments
- Tokenizer/model initialization
- Dataset loading and formatting

## 🚀 Training

### For LLaMA 3 (Unsloth)

```bash
python train_test_LLM.py
```

- Uses LoRA-based fine-tuning.
- Supports instruction-based datasets with `input`, `instruction`, and `output` fields.
- Automatically saves the fine-tuned model and tokenizer.

### For BERT/RoBERTa

```bash
python train_test_SLM.py
```

- Uses Hugging Face `Trainer` for classification fine-tuning.
- Trains on `primary_dataset` (and `secondary_dataset`, if defined).

## 🧪 Evaluation

Both `train_test_LLM.py` and `train_test_SLM.py` include automatic evaluation on:

- Circa test set
- SST2 validation set
- IMDB test set
- Amazon Polarity test set

The results include **accuracy** and **weighted F1**, saved to:

```bash
{exp_path}/[Pre-trained|Fine-tuned]evaluation_results.csv
```

## 📦 Dependencies

Install dependencies via pip:

```bash
pip install transformers datasets scikit-learn pandas unsloth tqdm
```

> You may need a CUDA-enabled GPU for training LLaMA models using Unsloth.

## 📁 Output

- Models are saved under `exp_path` based on model name and data paradigm.
- Evaluation metrics are written to a CSV file with one row per experiment.


## 🤝 Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Circa Dataset](https://github.com/circa)
