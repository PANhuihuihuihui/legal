from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_from_disk
import string
import re



#example = "(ii)   If so, whether it was reasonable for P to continue its action against D1 after receiving the Joint Report and/or after the withdrawal of contribution notice against D1 by D2?"



def clean_data(example):
    tmp = re.sub(r'^(\d+).', '', example)
    tmp = re.sub(r'^\((ix|iv|v?i{0,3})\)', '', tmp)
    tmp = re.sub(r'^\((\d+)\)', '', tmp)
    new_str = tmp.translate(str.maketrans('','',string.punctuation)).replace("’s",'').replace("s’",'s')
    return new_str

# date prepared for training



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
    tokenized_datasets=load_from_disk("/home/huijie/legal/huggface/data_prepare/HKLII_all_cail/" )

    # tokenized_datasets = hklii_dataset.map(
    #     tokenize_function, batched=True, remove_columns=["ID","topic","paragraphs"],num_proc =16
    # )
    hklii_dataset = tokenized_datasets.map(group_texts, batched=True,num_proc = 16)

    # train_size = 10_000
    # test_size = int(0.1 * train_size)

    # hklii_dataset = hklii_dataset["train"].train_test_split(
    #     train_size=train_size, test_size=test_size, seed=42
    # )


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
    eval_dataset =  eval_dataset.remove_columns(["masked_token_type_ids"])
    return hklii_dataset,eval_dataset



def get_element_datasets(args):
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    def tokenize_function(examples):
        result = tokenizer(examples["sentence"],truncation = True,max_length = 256)
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    dataset = load_dataset("json", data_files="/home/huijie/legal/mymodel/data/legal_element.json")
    dataset = dataset.shuffle(seed=args.seed).rename_columns("lables","classification").train_test_split(0.1)
    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["class","sentence",],num_proc =16
    )
    data_collator = DataCollatorWithPadding(tokenizer,padding= "max_length",max_length = 256,pad_to_multiple_of= 8),
    return tokenized_datasets["train"],tokenized_datasets["test"],data_collator


    
def get_scm_datasets(args):
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    def tokenize_function(examples):
        result = tokenizer(examples["sentence"],truncation = True,max_length = 512)
        if tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    dataset = load_dataset("json", data_files="/home/huijie/legal/mymodel/data/legal_element.json")
    dataset = dataset.shuffle(seed=args.seed).rename_columns("lables","classification").train_test_split(0.1)
    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["class","sentence",],num_proc =16
    )
    data_collator = DataCollatorWithPadding(tokenizer,padding= "max_length",max_length = 512,pad_to_multiple_of= 8),
    return tokenized_datasets["train"],tokenized_datasets["test"],data_collator




