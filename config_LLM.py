from transformers import TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
import torch
import os


class Config:
    def __init__(self):
        # Set model and data paradigm
        self.model_name = 'Llama3'
        self.data_paradigm = 'Questions and Answers'

        # Define experiment name and paths
        self.exp_name = f'{self.model_name}_{self.data_paradigm}'
        self.primary_dataset_name = 'Circa'
        self.secondary_dataset_name = 'None'
        self.base_path = '/dt/shabtaia/dt-fujitsu-LLMVisibility/Yarin'
        self.exp_path = f'{self.base_path}/{self.primary_dataset_name}/{self.secondary_dataset_name}/{self.exp_name}'

        self.llm_flow = 'Fine tuned'
        self.weight_dir = "unsloth/llama-3-8b-Instruct-bnb-4bit"

        # Adjust weight_dir if primary model was already trained
        if self.secondary_dataset_name != 'None' and os.path.exists(
                f'{self.base_path}/{self.primary_dataset_name}/None/{self.exp_name}'):
            self.weight_dir = os.path.exists(f'{self.base_path}/{self.primary_dataset_name}/None/{self.exp_name}')

        self.seed = 42
        self.tokenizer = None
        self.model = None
        self.primary_dataset = None
        self.secondary_dataset = None
        self.max_seq_length = 4096
        self.load_in_4bit = True
        self.dtype = None
        self.circa_dataset = None

        # Load data and initialize training arguments
        self.load_data()
        self.training_args = TrainingArguments(
            output_dir="/dt/shabtaia/dt-fujitsu-LLMVisibility/Yarin/results",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            logging_steps=10,
            num_train_epochs=1,
            save_strategy="epoch",
            eval_strategy="no",
            fp16=not torch.cuda.is_bf16_supported(),
            report_to="none"
        )

    def load_data(self):
        """
        Loads and prepares the Circa dataset, and optionally loads other datasets.
        """
        dataset = load_circa(self.data_paradigm)
        self.circa_dataset = dataset.map(lambda x: format_dataset(x, text_column='sentence', dataset_name='Circa'))

        primary_path = os.path.join(self.base_path, self.primary_dataset_name, "None", self.exp_name)
        if not os.path.exists(primary_path):
            self.primary_dataset = self._load_and_tokenize_dataset(self.primary_dataset_name)

        if not os.path.exists(self.exp_path):
            self.secondary_dataset = self._load_and_tokenize_dataset(self.secondary_dataset_name)

    def _load_and_tokenize_dataset(self, name):
        """
        Loads and tokenizes a dataset by name.

        Args:
            name (str): Dataset name (e.g., 'IMDB', 'SST2', 'AMAZON').

        Returns:
            DatasetDict: Tokenized dataset with train/validation split.
        """
        dataset_name_dict = {'IMDB': 'imdb', 'SST2': ("glue", "sst2"), "AMAZON": 'amazon_polarity'}
        dataset_name = dataset_name_dict[name]
        text_column_dict = {'imdb': 'text', ("glue", "sst2"): 'sentence', 'amazon_polarity': 'content'}
        text_column = text_column_dict[dataset_name]

        if name == 'Circa':
            return self.circa_dataset

        if isinstance(dataset_name, tuple):
            dataset = load_dataset(*dataset_name, split='train')
        else:
            dataset = load_dataset(dataset_name, split='train')

        dataset = dataset.map(lambda x: format_dataset(x, text_column=text_column, dataset_name=dataset_name))
        dataset = dataset.train_test_split(test_size=0.2, seed=42)

        return dataset


def load_circa(data_paradigm):
    """
    Loads and formats the Circa dataset according to the selected data paradigm.

    Args:
        data_paradigm (str): One of ['Questions and Answers', 'Answers only', 'Questions only'].

    Returns:
        DatasetDict: Train/validation/test split of the Circa dataset.
    """
    df = pd.read_csv("/sise/home/yarinye/fujitsu_WP3/CSNLU_Project/circa-data.tsv", sep='\t')

    # Construct input sentence based on the paradigm
    if data_paradigm == 'Questions and Answers':
        df['sentence'] = df['question-X']
    elif data_paradigm == 'Answers only':
        df['sentence'] = df['answer-Y']
    else:
        df['sentence'] = df.apply(lambda x: x['question-X'] + ' ' + x['answer-Y'], axis=1)

    # Map gold labels to binary format
    label_dict = {'Yes': 'Yes', 'No': 'No'}
    df['label'] = df['goldstandard2'].apply(lambda x: label_dict.get(x, None))
    df = df[['sentence', 'label']]
    df.dropna(inplace=True)

    # Split into train, val, test
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })


def format_dataset(example, text_column, dataset_name=''):
    """
    Formats an example from a dataset into an instruction-based prompt.

    Args:
        example (dict): A single example from the dataset.
        text_column (str): Column name containing the input text.
        dataset_name (str): Name of the dataset ('Circa', 'IMDB', etc.).

    Returns:
        dict: Dictionary with 'instruction', 'input', and 'output' fields.
    """
    if dataset_name == 'Circa':
        system_prompt = """You are a helpful, honest, and knowledgeable assistant. Your task is to analyze a user's answer and predict the underlying intention — 
            whether the user means 'Yes' or 'No'. Focus only on the intention behind the message, even if it's implicit, sarcastic, or indirect. 
            Respond only with either 'Yes' or 'No'. If the intention is unclear, choose the dominant option. Be consistent, accurate, and do not guess."""
        return {
            "instruction": system_prompt,
            "input": example[text_column],
            "output": example["label"]
        }
    else:
        system_prompt = """You are a helpful, honest, and knowledgeable assistant. Your task is to analyze the sentiment of a user's message 
            and classify it strictly as either 'Positive' or 'Negative'. Focus only on the overall emotional tone. 
            Do not include explanations or additional output. If sentiment is mixed or unclear, choose the dominant sentiment."""

        return {
            "instruction": system_prompt,
            "input": example[text_column],
            "output": "positive" if example["label"] == 1 else "negative"
        }


def tokenize_dataset(dataset, tokenizer, max_length):
    """
    Tokenizes a dataset using a conversational prompt format for chat-based LLMs.

    Args:
        dataset (DatasetDict): Dataset containing 'instruction', 'input', and 'output'.
        tokenizer: HuggingFace tokenizer with chat template support.
        max_length (int): Maximum sequence length.

    Returns:
        DatasetDict: Tokenized dataset.
    """

    def tokenize(example):
        # Convert instruction + input + output into chat format
        chat = [
            {"role": "user", "content": f"{example['instruction']}\n{example['input']}"},
            {"role": "assistant", "content": example["output"]}
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return tokenizer(prompt, truncation=True, max_length=max_length, padding=False)

    return dataset.map(tokenize, remove_columns=dataset.column_names)
