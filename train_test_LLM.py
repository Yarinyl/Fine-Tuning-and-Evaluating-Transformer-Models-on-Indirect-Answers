from config_LLM import Config, tokenize_dataset
import csv
from tqdm import tqdm
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import numpy as np
import torch
from transformers import EvalPrediction
from trl import SFTTrainer
from unsloth import FastLanguageModel
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
import os


def compute_metrics(p: EvalPrediction):
    """
    Computes classification metrics for model evaluation.

    Args:
        p (EvalPrediction): Evaluation predictions and labels.

    Returns:
        dict: Accuracy and weighted F1 score.
    """
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average="weighted")
    }


def train(config):
    """
    Fine-tunes a LLaMA-3 model using the provided configuration.

    Args:
        config (Config): Configuration object containing datasets, model paths, and training args.
    """
    print('Start Training')
    print(config.exp_path)

    dataset = config.primary_dataset

    # Load base LLaMA model with tokenizer
    print("Loading LLaMA 3 model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.weight_dir,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )

    # Apply LoRA fine-tuning to reduce memory usage
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Tokenize training dataset
    print("Tokenizing dataset...")
    tokenized_dataset = {
        "train": tokenize_dataset(dataset["train"], tokenizer, config.max_seq_length)
    }

    # Initialize supervised fine-tuning trainer
    trainer = SFTTrainer(
        model=model,
        args=config.training_args,
        train_dataset=tokenized_dataset["train"],
        compute_metrics=compute_metrics,
    )

    # Start training
    print("Starting training...")
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    trainer.train()

    # Save the trained model and tokenizer
    print("Saving model...")
    trainer.save_model(config.exp_path)
    tokenizer.save_pretrained(config.exp_path)
    print("Training complete.")


def test(config):
    """
    Evaluates the fine-tuned or pre-trained model on multiple test sets and saves the results.

    Args:
        config (Config): Configuration object used during training.
    """
    print("Start Testing")
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    print(config.exp_path)

    # Load model: pre-trained or fine-tuned version
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-Instruct-bnb-4bit" if config.llm_flow == 'Pre-trained' else config.exp_path,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )
    model.eval()

    def eval_llm(dataset, positive_word="positive", bs=64):
        """
        Performs evaluation by generating predictions and matching keyword.

        Args:
            dataset (Dataset): The tokenized test dataset.
            positive_word (str): The word to check for positive prediction.
            bs (int): Batch size.

        Returns:
            dict: Accuracy and F1 score.
        """
        data_collator = DataCollatorWithPadding(tokenizer, padding="longest")
        all_labels = dataset["label"]
        dataset = tokenize_dataset(dataset, tokenizer, config.max_seq_length)
        loader = DataLoader(dataset, batch_size=bs, collate_fn=data_collator)

        all_preds = []
        with torch.inference_mode():
            for batch in tqdm(loader):
                batch = {k: v.to('cuda') for k, v in batch.items()}
                generated = model.generate(
                    **batch,
                    max_new_tokens=5,
                    eos_token_id=tokenizer.eos_token_id,
                )
                prompt_len = batch["input_ids"].size(1)
                new_ids = generated[:, prompt_len:]
                texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)

                for txt in texts:
                    pred = 1 if positive_word in txt.lower() else 0
                    all_preds.append(pred)

        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds)
        }

    # Run evaluation on Circa
    circa_results = eval_llm(config.circa_dataset["test"], positive_word='yes')
    print("Circa Test Results:", circa_results)

    # Run evaluation on SST-2
    sst2 = load_dataset("glue", "sst2", split="validation")
    sst2 = sst2.map(lambda x: format_dataset(x, text_column="sentence"))
    sst2_results = eval_llm(sst2)
    print("SST2 Test Results:", sst2_results)

    # Run evaluation on IMDB
    imdb = load_dataset("imdb", split="test")
    imdb = imdb.map(lambda x: format_dataset(x, text_column="text"))
    imdb_results = eval_llm(imdb, bs=4)
    print("IMDB Test Results:", imdb_results)

    # Run evaluation on Amazon Polarity
    amazon = load_dataset("amazon_polarity", split="test")
    amazon = amazon.map(lambda x: format_dataset(x, text_column="content"))
    amazon_results = eval_llm(amazon, bs=4)
    print("Amazon Test Results:", amazon_results)

    def extract_metrics(results):
        """
        Helper function to extract accuracy and F1 from a result dict.
        """
        return [results.get("accuracy", 0), results.get("f1", 0)]

    row = (
        extract_metrics(circa_results) +
        extract_metrics(sst2_results) +
        extract_metrics(imdb_results) +
        extract_metrics(amazon_results)
    )

    # Define CSV headers
    headers = [
        "Indirect Answers Accuracy", "Indirect Answers F1",
        "SST2 Accuracy", "SST2 F1",
        "IMDB Accuracy", "IMDB F1",
        "Amazon Accuracy", "Amazon F1"
    ]

    # Write evaluation results to CSV
    with open(f"{config.exp_path}/{config.llm_flow if config.llm_flow is not None else ''}evaluation_results.csv",
              mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(row)


if __name__ == '__main__':
    conf = Config()
    if not os.path.exists(conf.exp_path):
        train(conf)
        conf = Config()
    test(conf)
