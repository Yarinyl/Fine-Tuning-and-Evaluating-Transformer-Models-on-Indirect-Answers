import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, EvalPrediction, TrainingArguments
from transformers import AutoModelForSequenceClassification
from config_SLM import Config, tokenize_sentiment_dataset
import csv
import os
from datasets import load_dataset


def compute_metrics(p: EvalPrediction):
    """
    Compute accuracy and weighted F1 score for model predictions.

    Args:
        p (EvalPrediction): Evaluation object containing predictions and labels.

    Returns:
        dict: Dictionary with accuracy and F1 score.
    """
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average="weighted")
    }


def train(config):
    """
    Fine-tunes a transformer-based sequence classification model
    using one or two datasets defined in the configuration.

    Args:
        config (Config): Configuration object with model, tokenizer, datasets, and training args.
    """
    print('Start Training')
    print(config.exp_path)

    # Train on primary dataset (if exists)
    if config.primary_dataset:
        trainer = Trainer(
            model=config.model,
            args=config.training_args,
            train_dataset=config.primary_dataset["train"],
            eval_dataset=config.primary_dataset["validation"],
            compute_metrics=compute_metrics,
        )
        trainer.train()

    # Train on secondary dataset (if exists)
    if config.secondary_dataset:
        trainer = Trainer(
            model=config.model,
            args=config.training_args,
            train_dataset=config.secondary_dataset["train"],
            eval_dataset=config.secondary_dataset["validation"],
            compute_metrics=compute_metrics,
        )
        trainer.train()

    # Save model and tokenizer
    trainer.save_model(config.exp_path)
    config.tokenizer.save_pretrained(config.exp_path)


def test(config):
    """
    Evaluates the best saved model on multiple benchmark datasets and saves results to CSV.

    Args:
        config (Config): Configuration object including dataset, tokenizer, and evaluation parameters.
    """
    print("Start Testing")

    # Load best model
    best_model = AutoModelForSequenceClassification.from_pretrained(config.exp_path)
    config.training_args.eval_strategy = "no"

    best_trainer = Trainer(
        model=best_model,
        args=config.training_args,
        compute_metrics=compute_metrics,
    )

    print(config.exp_path)

    # Evaluate on Circa test set
    circa_results = best_trainer.evaluate(eval_dataset=config.circa_dataset["test"])
    print("Circa Test Results:", circa_results)

    # Load and evaluate SST2
    sst2 = load_dataset("glue", "sst2")["validation"]
    sst2 = tokenize_sentiment_dataset(sst2, config.tokenizer, text_column="sentence", label_column="label")

    # Load and evaluate IMDB
    imdb = load_dataset("imdb")["test"]
    imdb = tokenize_sentiment_dataset(imdb, config.tokenizer, text_column="text", label_column="label")

    # Load and evaluate Amazon Polarity
    amazon = load_dataset("amazon_polarity", split='test')
    amazon = tokenize_sentiment_dataset(amazon, config.tokenizer, text_column="content", label_column="label")

    # Evaluate and print results
    sst2_results = best_trainer.evaluate(eval_dataset=sst2)
    print("SST2 Test Results:", sst2_results)

    imdb_results = best_trainer.evaluate(eval_dataset=imdb)
    print("IMDB Test Results:", imdb_results)

    amazon_results = best_trainer.evaluate(eval_dataset=amazon)
    print("Amazon Test Results:", amazon_results)

    def extract_metrics(results):
        """
        Extract accuracy and F1 score from result dict.

        Args:
            results (dict): Evaluation results.

        Returns:
            list: [accuracy, f1]
        """
        return [results.get("accuracy", 0), results.get("f1", 0)]

    # Combine all evaluation metrics
    row = (
        extract_metrics(circa_results) +
        extract_metrics(sst2_results) +
        extract_metrics(imdb_results) +
        extract_metrics(amazon_results)
    )

    # CSV column headers
    headers = [
        "Indirect Answers Accuracy", "Indirect Answers F1",
        "SST2 Accuracy", "SST2 F1",
        "IMDB Accuracy", "IMDB F1",
        "Amazon Accuracy", "Amazon F1"
    ]

    # Save results to CSV
    with open(f"{config.exp_path}/evaluation_results.csv",
              mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(row)


if __name__ == '__main__':
    conf = Config()
    # Only train if the model is not already saved
    if not os.path.exists(conf.exp_path):
        train(conf)

    test(conf)
