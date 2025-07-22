from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    TrainingArguments, AutoModelForSequenceClassification
)
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
import os


class Config:
    """
    Configuration class for setting up model, tokenizer, dataset, and training arguments
    for BERT or RoBERTa fine-tuning on classification tasks.
    """

    def __init__(self):
        # Model and data selection
        self.model_name = 'BERT'  # Options: 'BERT', 'RoBERTa'
        self.data_paradigm = 'Questions and Answers'  # Options: 'Questions only', 'Answers only', 'Questions and Answers'
        self.exp_name = f'{self.model_name}_{self.data_paradigm}'
        self.primary_dataset_name = 'Circa'  # Options: 'Circa', 'IMDB', 'SST2', 'Amazon'
        self.secondary_dataset_name = 'None'  # Can be another dataset name or 'None'

        # Directory paths
        self.base_path = '/dt/shabtaia/dt-fujitsu-LLMVisibility/Yarin'
        self.exp_path = f'{self.base_path}/{self.primary_dataset_name}/{self.secondary_dataset_name}/{self.exp_name}'

        # Other settings
        self.seed = 42
        self.tokenizer = None
        self.model = None
        self.primary_dataset = None
        self.secondary_dataset = None
        self.circa_dataset = None

        # Initialize model, tokenizer, and datasets
        self.load_models()
        self.load_data()

        # Define Hugging Face training arguments
        self.training_args = TrainingArguments(
            output_dir="/dt/shabtaia/dt-fujitsu-LLMVisibility/Yarin/results",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=2,
        )

    def load_models(self):
        """
        Loads the tokenizer and model for BERT or RoBERTa.
        If a fine-tuned model exists at the expected path, it is loaded;
        otherwise, a new pre-trained model is initialized with 2 output labels.
        """
        if self.model_name == 'BERT':
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            if os.path.exists(f'{self.base_path}/{self.primary_dataset_name}/None/{self.exp_name}'):
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    f'{self.base_path}/{self.primary_dataset_name}/None/{self.exp_name}/')
            else:
                self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

        elif self.model_name == 'RoBERTa':
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            if os.path.exists(f'{self.base_path}/{self.primary_dataset_name}/None/{self.exp_name}'):
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    f'{self.base_path}/{self.primary_dataset_name}/None/{self.exp_name}/')
            else:
                self.model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    def load_data(self):
        """
        Loads and tokenizes the Circa dataset and other specified datasets (if needed).
        Applies padding and truncation to ensure compatibility with the transformer model.
        """
        def tokenize(batch):
            return self.tokenizer(batch["sentence"], padding="max_length", truncation=True)

        # Load and process Circa
        dataset = load_circa(self.data_paradigm)
        tokenized_dataset = dataset.map(tokenize, batched=True)
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        self.circa_dataset = tokenized_dataset

        # Load additional datasets if applicable
        primary_path = os.path.join(self.base_path, self.primary_dataset_name, "None", self.exp_name)
        if not os.path.exists(primary_path):
            self.primary_dataset = self._load_and_tokenize_dataset(self.primary_dataset_name)

        if not os.path.exists(self.exp_path):
            self.secondary_dataset = self._load_and_tokenize_dataset(self.secondary_dataset_name)

    def _load_and_tokenize_dataset(self, name):
        """
        Loads and tokenizes a sentiment analysis dataset such as SST2, IMDB, or Amazon.

        Args:
            name (str): Name of the dataset to load.

        Returns:
            DatasetDict: Tokenized and split dataset.
        """
        if name == 'Circa':
            return self.circa_dataset

        if name == 'SST2':
            ds = load_dataset("glue", "sst2")["train"]
            ds = tokenize_sentiment_dataset(ds, self.tokenizer, text_column="sentence", label_column="label")
        elif name == 'IMDB':
            ds = load_dataset("imdb", split="train")
            ds = tokenize_sentiment_dataset(ds, self.tokenizer, text_column="text", label_column="label")
        elif name == 'Amazon':
            ds = load_dataset("amazon_polarity")["train"]
            ds = tokenize_sentiment_dataset(ds, self.tokenizer, text_column="content", label_column="label")

        ds = ds.train_test_split(test_size=0.2, seed=self.seed)
        return DatasetDict({
            "train": ds["train"],
            "validation": ds["test"]
        })


def load_circa(data_paradigm):
    """
    Loads the Circa dataset and formats it based on the specified paradigm.

    Args:
        data_paradigm (str): One of ['Questions and Answers', 'Questions only', 'Answers only'].

    Returns:
        DatasetDict: Split dataset into train, validation, and test sets.
    """
    df = pd.read_csv("/sise/home/yarinye/fujitsu_WP3/CSNLU_Project/circa-data.tsv", sep='\t')

    # Construct input sentence column
    if data_paradigm == 'Questions and Answers':
        df['sentence'] = df['question-X']
    elif data_paradigm == 'Answers only':
        df['sentence'] = df['answer-Y']
    else:
        df['sentence'] = df.apply(lambda x: x['question-X'] + ' ' + x['answer-Y'], axis=1)

    # Binary label mapping
    label_dict = {'Yes': 'Yes', 'No': 'No'}
    df['label'] = df['goldstandard2'].apply(lambda x: label_dict.get(x, None))
    df = df[['sentence', 'label']]
    df.dropna(inplace=True)

    # Split into train/validation/test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })


def tokenize_sentiment_dataset(dataset, tokenizer, text_column="text", label_column="label"):
    """
    Tokenizes a sentiment analysis dataset and formats it for training.

    Args:
        dataset (Dataset): Hugging Face dataset object.
        tokenizer (Tokenizer): Hugging Face tokenizer.
        text_column (str): Name of the column containing text.
        label_column (str): Name of the column containing labels.

    Returns:
        Dataset: Tokenized dataset ready for use with transformers.
    """
    dataset = dataset.map(lambda x: tokenizer(x[text_column], truncation=True, padding="max_length"), batched=True)
    dataset = dataset.rename_column(label_column, "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset
