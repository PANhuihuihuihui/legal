from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import string
import re
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")


# filter the length < 5 
def clean_data(example):
    tmp = re.sub(r'^(\d+).', '', example)
    tmp = re.sub(r'^\((ix|iv|v?i{0,3})\)', '', tmp)
    tmp = re.sub(r'^\((\d+)\)', '', tmp)
    new_str = tmp.translate(str.maketrans('','',string.punctuation)).replace("’s",'').replace("s’",'s')
    return new_str

# date prepared for training
def tokenize_function(examples):
    lower = [clean_data(x.lower()) for x in examples["paragraphs"]]
    # lower = filter(lambda x : len(x)>7, lower)
    # print(lower)
    result = tokenizer(lower)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


hklii_dataset_en = load_dataset("json", data_files="/home/huijie/legal/huggface/data_prepare/HKLII-online/HKLII.json")
hklii_dataset_en = hklii_dataset_en["train"].remove_columns("topic")
print(hklii_dataset_en)

hklii_dataset_ch = load_dataset("json", data_files="/home/xijia/nlp/data_prepare/data/HKLII_zh.json")
hklii_dataset_ch = hklii_dataset_ch["train"].remove_columns("topic")
print(hklii_dataset_ch)

cail_dataset_ch = load_dataset("json",data_files = "/home/xijia/nlp/cail_data_prepare/final_test.json")
cail_dataset_ch = cail_dataset_ch["train"].remove_columns("topic")
print(cail_dataset_ch)


assert hklii_dataset_en.features.type == hklii_dataset_ch.features.type
assert hklii_dataset_en.features.type == cail_dataset_ch.features.type
hklii_dataset = concatenate_datasets([hklii_dataset_en, hklii_dataset_ch,cail_dataset_ch])
print(hklii_dataset)


hklii_dataset = hklii_dataset.train_test_split(test_size=0.1)

tokenized_datasets = hklii_dataset.map(
    tokenize_function, batched=True, remove_columns=["ID","paragraphs"],num_proc = 8)
print(tokenized_datasets)

tokenized_datasets = tokenized_datasets.filter(lambda example: len(example['input_ids']) > 7,num_proc=8)
print(tokenized_datasets)

stat = [len(inp) for inp in tokenized_datasets["train"]["input_ids"]]

s = pd.Series(stat)
print(s.describe())

# count    3.036218e+06
# mean     7.891106e+01
# std      7.347223e+01
# min      2.000000e+00
# 25%      2.900000e+01
# 50%      6.000000e+01
# 75%      1.080000e+02
# max      1.181900e+04
# dtype: float64


tokenized_datasets.save_to_disk("/home/huijie/legal/huggface/data_prepare/HKLII_all_cail")
