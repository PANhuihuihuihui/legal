# get all the file 
import os
import json
from tqdm import tqdm
import itertools
folder_list = os.listdir("/home/xijia/nlp/log")
path=  "/home/xijia/nlp/log/{}/data/"
exclude_list = ["parties","judgment","references"]

for topic in folder_list:
    topic_path = path.format(topic)
    files = os.listdir(topic_path)
    for file in tqdm(files):
        dic = {}
        with open(os.path.join(topic_path,file)) as json_data:
            json_data = json.load(json_data)
            value_list = [json_data[k] for k in json_data.keys() if k not in exclude_list]
            value_list = list(itertools.chain(*value_list))
            dic= {
                "file_path": json_data["file_path"],
                "paragraphs": value_list
            }
        target_path = "/home/huijie/legal/huggface/data_prepare/data/{}".format(topic)
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        with open("/home/huijie/legal/huggface/data_prepare/data/{}/{}".format(topic,file), 'w+') as json_file:
            json.dump(dic, json_file)
        









# build hugface datasets
# from datasets import load_dataset

# hklii_dataset = load_dataset("json", data_files="/home/xijia/nlp/log/HKDC/data/*.json", field="judgment")
# print(hklii_dataset)
