from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import DataCollatorForLanguageModeling
from datasets import load_from_disk
import config


tokenizer = AutoTokenizer.from_pretrained("zlucia/custom-legalbert")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


# date prepared for training
def tokenize_function(examples):
    lower = [x.lower() for x in examples["paragraphs"]]
    result = tokenizer(lower)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}


def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // config.chunk_size) * config.chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + config.chunk_size] for i in range(0, total_length, config.chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result




## data loader
def getDataloader(hklii_dataset,eval_dataset):
    train_dataloader = DataLoader(
        hklii_dataset["train"],
        shuffle=True,
        batch_size=config.batch_size,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=config.batch_size, collate_fn=default_data_collator
    )
    return train_dataloader,eval_dataloader

def getDataset():
    hklii_dataset=load_from_disk("/home/huijie/legal/huggface/data_prepare/HKLII/" )

    tokenized_datasets = hklii_dataset.map(
        tokenize_function, batched=True, remove_columns=["ID","topic","paragraphs"],num_proc =16
    )
    hklii_dataset = tokenized_datasets.map(group_texts, batched=True,num_proc = 16)

    hklii_dataset = hklii_dataset.remove_columns(["word_ids"])

    eval_dataset = hklii_dataset["test"].map(
        insert_random_mask,
        batched=True,
        remove_columns=hklii_dataset["test"].column_names,
    )
    eval_dataset = eval_dataset.rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
        }
    )
    return hklii_dataset,eval_dataset


