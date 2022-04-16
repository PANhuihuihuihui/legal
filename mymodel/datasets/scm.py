
import json
import logging
import os
import random
from typing import Tuple, List, Union


import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data.dataloader import default_collate



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

