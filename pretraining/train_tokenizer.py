from tqdm import tqdm
from transformers import GPT2TokenizerFast
from datasets import load_dataset
import os

VOCAB_SIZE = 50257
MAX_LENGTH = 1024
SEED = 42
MODEL_NAME = "gpt-2"

# Load the OpenWebText dataset
raw_datasets = load_dataset("openwebtext", split="train")
# Split the dataset into train and test sets (1% for test)
raw_datasets = raw_datasets.train_test_split(test_size=0.01)
raw_train_dataset = raw_datasets["train"]
raw_valid_dataset = raw_datasets["test"]

# Define an iterator for batch loading data
def batch_iterator(batch_size=10000):
    for i in tqdm(range(0, len(raw_train_dataset), batch_size)):
        yield raw_train_dataset[i: i + batch_size]["text"]

# Load the pre-trained tokenizer (to reuse its defined special tokens)
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Train a new tokenizer from the iterator
gpt_tokenizer = tokenizer.train_new_from_iterator(
    text_iterator=batch_iterator(), vocab_size=VOCAB_SIZE)

# Save the new tokenizer
gpt_tokenizer.save_pretrained("my-tokenizer-"+MODEL_NAME)
