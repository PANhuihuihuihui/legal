
import json
import logging
import os
import random
from typing import Tuple, List, Union

from torch.utils.data.dataloader import default_collate

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    BertConfig,
    BertTokenizer,
)
from transformers import BertPreTrainedModel, BertModel




class TripletTextDataset(Dataset):
    def __init__(self, text_a_list, text_b_list, text_c_list, label_list=None):
        if label_list is None or len(label_list) == 0:
            label_list = [None] * len(text_a_list)
        assert all(
            len(label_list) == len(text_list)
            for text_list in [text_a_list, text_b_list, text_c_list]
        )
        self.text_a_list = text_a_list
        self.text_b_list = text_b_list
        self.text_c_list = text_c_list
        self.label_list = [0 if label == "B" else 1 for label in label_list]

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        text_a, text_b, text_c, label = (
            self.text_a_list[index],
            self.text_b_list[index],
            self.text_c_list[index],
            self.label_list[index],
        )
        return text_a, text_b, text_c, label

    @classmethod
    def from_dataframe(cls, df):
        text_a_list = df["A"].tolist()
        text_b_list = df["B"].tolist()
        text_c_list = df["C"].tolist()
        if "label" not in df:
            df["label"] = "B"
        label_list = df["label"].tolist()
        return cls(text_a_list, text_b_list, text_c_list, label_list)

    @classmethod
    def from_dict_list(cls, data, use_augment=False):
        df = pd.DataFrame(data)
        if "label" not in df:
            df["label"] = "B"
        if use_augment:
            df = TripletTextDataset.augment(df)
        return cls.from_dataframe(df)

    @classmethod
    def from_jsons(cls, json_lines_file, use_augment=False):
        with open(json_lines_file, encoding="utf-8") as f:
            data = list(map(lambda line: json.loads(line), f))
        return cls.from_dict_list(data, use_augment)

    @staticmethod
    def augment(df):
        # 反对称增广
        df_cp1 = df.copy()
        df_cp1["B"] = df["C"]
        df_cp1["C"] = df["B"]
        df_cp1["label"] = df_cp1["label"].apply(
            lambda label: "C" if label == "B" else "B"
        )

        # 自反性增广
        df_cp2 = df.copy()
        df_cp2["A"] = df["C"]
        df_cp2["B"] = df["C"]
        df_cp2["C"] = df["A"]
        df_cp2["label"] = "B"

        # 自反性+反对称增广
        df_cp3 = df.copy()
        df_cp3["A"] = df["C"]
        df_cp3["B"] = df["A"]
        df_cp3["C"] = df["C"]
        df_cp3["label"] = "C"

        # 启发式增广
        df_cp4 = df.copy()
        df_cp4 = df_cp4.apply(
            lambda x: pd.Series((x["B"], x["A"], x["C"], "B"))
            if x["label"] == "B"
            else pd.Series((x["C"], x["B"], x["A"], "C")),
            axis=1,
            result_type="broadcast",
        )

        # 启发式+反对称增广
        df_cp5 = df.copy()
        df_cp5 = df_cp5.apply(
            lambda x: pd.Series((x["B"], x["C"], x["A"], "C"))
            if x["label"] == "B"
            else pd.Series((x["C"], x["A"], x["B"], "B")),
            axis=1,
            result_type="broadcast",
        )

        df = pd.concat([df, df_cp1, df_cp2, df_cp3, df_cp4, df_cp5])
        df = df.drop_duplicates()
        df = df.sample(frac=1)

        return df

def get_collator(max_len, device, tokenizer, model_class):
    def two_pair_collate_fn(batch):
        """
        获取一个mini batch的数据，将文本三元组转化成tensor。

        将ab、ac分别拼接，编码tensor

        :param batch:
        :return:
        """
        example_tensors = []
        for text_a, text_b, text_c, label in batch:
            input_example = InputExample(text_a, text_b, text_c, label)
            ab_feature, ac_feature = input_example.to_two_pair_feature(
                tokenizer, max_len
            )
            ab_tensor, ac_tensor = (
                ab_feature.to_tensor(device),
                ac_feature.to_tensor(device),
            )
            label_tensor = torch.LongTensor([label]).to(device)
            example_tensors.append((ab_tensor, ac_tensor, label_tensor))

        return default_collate(example_tensors)

    if model_class == BertForSimMatchModel:
        return two_pair_collate_fn

