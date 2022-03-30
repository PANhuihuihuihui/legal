# get all the file 

# expect format should be: {pargraph: filepath: } all in one json
import os
import json
from tqdm import tqdm
# import pandas as pd
import codecs
folder_list = os.listdir("/home/xijia/nlp/log")
path=  "/home/xijia/nlp/log3/{}/data/"
exclude_list = ["parties","references","file_path"]
len_status = []
for topic in folder_list:
    topic_path = path.format(topic)
    files = os.listdir(topic_path)
    for file in tqdm(files):
        dic = {}
        with codecs.open(os.path.join(topic_path,file),encoding="utf-8") as json_data:
            json_dic = json.load(json_data)
            value_list = []
            for k in json_dic["judgment"].keys():
                if k != "":
                    pargraph = list(filter(lambda x: len(x.split()) > 5 ,json_dic["judgment"][k]))
                    value_list.extend(json_dic["judgment"][k])
            
            len_status.append(len(value_list))
            
        if len_status[-1] == 0:
            continue # ignore empty judgement 
        target_path = "/home/huijie/legal/huggface/data_prepare/data/{}".format(topic)
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        with codecs.open("/home/huijie/legal/huggface/data_prepare/data/HKLII.json", 'a+') as json_file:
            for i in value_list:
                if len(i.split()) < 20:
                    continue   
                
                dic= {
                    "paragraphs": i,
                    "topic": topic,
                    "ID": file
                    }    
                json.dump(dic, json_file,ensure_ascii=False)
                json_file.write('\n')
# s = pd.Series(len_status)
# print(s.describe())
# this is the length of the paragraph 
# count    80869.000000
# mean        49.203044
# std         54.510695
# min          0.000000
# 25%         21.000000
# 50%         34.000000
# 75%         58.000000
# max       2271.000000

# words of the chinese
# count    1.458006e+06
# mean     7.819433e+01
# std      9.874830e+01
# min      1.000000e+00
# 25%      1.500000e+01
# 50%      4.400000e+01
# 75%      1.080000e+02
# max      6.492000e+03
    
# build hugface datasets
from datasets import load_dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("zlucia/custom-legalbert")
# processin funtion
def lowercase_condition(example):
    return {"condition": example["paragraphs"].lower()}

def filter_nones(x):
    return x["condition"] is not None
def filter_sentence_length(example):
    example["paragraphs"] = filter(lambda x: len(x.split()) > 5 ,example["paragraphs"])
    return example

def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=512,
        return_overflowing_tokens=True,
    )

folder_list = os.listdir("/home/xijia/nlp/log")
data_files = {k:"/home/huijie/legal/huggface/data_prepare/data/{}/*.json".format(k) for k in folder_list}
hklii_dataset = load_dataset("json", data_files=data_files)
print(hklii_dataset)

print(hklii_dataset['paragraphs'][:3])

# filer the sentence length
hklii_dataset = hklii_dataset.map(filter_sentence_length,num_proc=4,batched=True)
print(hklii_dataset['paragraphs'][:3])


result = tokenize_and_split(drug_dataset["train"][0])













