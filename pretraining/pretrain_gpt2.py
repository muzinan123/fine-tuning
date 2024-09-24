import torch
import torch.distributed
from datasets import load_dataset
import os
from transformers import GPT2TokenizerFast, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
import multiprocessing
import random

MODEL_NAME = "gpt2"
SEED = 42
MAX_LENGTH = 1024
MIN_LENGTH = 4
VOCAB_SIZE = 50257

# Define data processing function
def prepare_train_features(examples):
    # Skip empty lines
    examples["nonempty_text"] = [
        d.strip() for d in examples["text"] if len(d.strip()) > 0
    ]

    # Convert the tokens into ids using the trained tokenizer
    tokenized_example = tokenizer(
        examples["nonempty_text"],
        truncation=True,
        max_length=MAX_LENGTH*100,
    )

    # Fields for model output
    examples["input_ids"] = []
    examples["attention_mask"] = []

    del examples["text"]
    del examples["nonempty_text"]

    for input_ids, attention_mask in zip(tokenized_example["input_ids"], tokenized_example["attention_mask"]):
        trunc_ids = input_ids[:min(len(input_ids), MAX_LENGTH)]
        trunc_mask = attention_mask[:min(len(attention_mask), MAX_LENGTH)]

        # Split long sentences into MAX_LENGTH segments, ignore the last segment if it's smaller than MIN_LENGTH
        while len(trunc_ids) > MIN_LENGTH:
            trunc_len = len(trunc_ids)
            if trunc_len < MAX_LENGTH:
                examples["input_ids"].append(
                    trunc_ids+[tokenizer.pad_token_id]*(MAX_LENGTH-trunc_len))
                examples["attention_mask"].append(
                    trunc_mask+[0]*(MAX_LENGTH-trunc_len))
            else:
                examples["input_ids"].append(trunc_ids)
                examples["attention_mask"].append(trunc_mask)

            input_ids = input_ids[trunc_len:]
            attention_mask = attention_mask[trunc_len:]

            trunc_ids = input_ids[:min(len(input_ids), MAX_LENGTH)]
            trunc_mask = attention_mask[:min(len(attention_mask), MAX_LENGTH)]

    examples['labels'] = examples['input_ids'].copy()

    return examples

# Enable distributed training mode
torch.distributed.init_process_group(
    backend='nccl', init_method="env://", rank=args.local_rank, world_size=args.word_size)
torch.cuda.set_device(args.local_rank)

# Automatically download the openwebtext dataset, which expands to about 500GB in arrow format
raw_datasets = load_dataset("openwebtext", split="train")
# Use only 1% of the data as the test set to reduce dev time
raw_datasets = raw_datasets.train_test_split(test_size=0.01)

raw_train_dataset = raw_datasets["train"]
raw_valid_dataset = raw_datasets["test"]

transformers.set_seed(args.seed)

# Define tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
# Get the number of CPU cores (for data loading threads)
num_proc = multiprocessing.cpu_count()

if torch.distributed.get_rank() > 0:
    # Main process loads data, other processes wait to load from cached arrow files
    torch.distributed.barrier()

tokenized_train_dataset = raw_train_dataset.map(
    prepare_train_features,
    batched=True,
    num_proc=num_proc
)

tokenized_valid_dataset = raw_valid_dataset.map(
    prepare_train_features,
    batched=True,
    num_proc=num_proc
)

if torch.distributed.get_rank() == 0:
    # Main process finishes loading data
    torch.distributed.barrier()

# Define data collator (automatically generate batches)
collater = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, return_tensors="pt"
)

# Define GPT-2 model config
model_config = GPT2Config(vocab_size=VOCAB_SIZE,
                          max_position_embeddings=MAX_LENGTH, return_dict=True)
# Define model (parameters are randomly initialized here)
model = GPT2LMHeadModel(config=model_config)

training_args = TrainingArguments(
    output_dir="./my_model",        # Path to save checkpoints
    evaluation_strategy="steps",    # Evaluate every N steps
    overwrite_output_dir=True,
    num_train_epochs=1,             # Number of training epochs
    per_device_train_batch_size=8,  # Batch size per GPU
    gradient_accumulation_steps=20, # Number of steps to accumulate gradients before updating parameters
    per_device_eval_batch_size=16,  # Evaluation batch size
    logging_steps=1000,             # Evaluate every 1000 steps
    save_steps=1000,                # Save checkpoint every 1000 steps
    learning_rate=1e-3,             # Learning rate
    warmup_steps=2000,              # Warmup steps (optional)
    optim="adamw_hf",               # Optimizer (default)
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collater,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
)

os.environ["WANDB_DISABLED"] = "true"

# Start training
trainer.train()
