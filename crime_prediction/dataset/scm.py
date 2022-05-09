
import json
from typing import Tuple, List, Union


import numpy as np
import pandas as pd
import torch

from torch.utils.data.dataloader import default_collate
# @staticmethod
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

if __name__ == '__main__':
    df = pd.read_json (r'/home/huijie/legal/cail_scm_2/data/raw/CAIL2019-SCM-big/SCM_5k.json',lines=True)
    print("length before :",len(df["B"]))
    stat_a = [len(inp) for inp in df["A"]]
    stat_b = [len(inp) for inp in df["B"]]
    stat_c = [len(inp) for inp in df["C"]]
    stat_label = [len(inp) for inp in df["label"]]
    print("A stat")
    s = pd.Series(stat_a)
    print(s.describe())
    print("A stat")
    s = pd.Series(stat_b)
    print(s.describe())
    print("C stat")
    s = pd.Series(stat_c)
    print(s.describe())
    print("label stat")
    s = pd.Series(stat_label)
    print(s.describe())


    df = augment(df)

    stat_a = [len(inp) for inp in df["A"]]
    stat_b = [len(inp) for inp in df["B"]]
    stat_c = [len(inp) for inp in df["C"]]
    stat_label = [len(inp) for inp in df["label"]]
    print("A stat")
    s = pd.Series(stat_a)
    print(s.describe())
    print("A stat")
    s = pd.Series(stat_b)
    print(s.describe())
    print("C stat")
    s = pd.Series(stat_c)
    print(s.describe())
    print("label stat")
    s = pd.Series(stat_label)
    print(s.describe())
    print("length after :",len(df["B"]))

    # with open('/home/huijie/legal/mymodel/data/SCM.json'', 'w', encoding='utf-8') as file:
    #     df.to_json(file, force_ascii=False)
    df.to_json(r'/home/huijie/legal/mymodel/data/SCM.json',orient='records',force_ascii=False,lines=True)

    df = pd.read_json (r'/home/huijie/legal/mymodel/data/SCM.json',lines=True)

    stat_a = [len(inp) for inp in df["A"]]
    stat_b = [len(inp) for inp in df["B"]]
    stat_c = [len(inp) for inp in df["C"]]
    stat_label = [len(inp) for inp in df["label"]]

    print("A stat")
    s = pd.Series(stat_a)
    print(s.describe())
    print("A stat")
    s = pd.Series(stat_b)
    print(s.describe())
    print("C stat")
    s = pd.Series(stat_c)
    print(s.describe())
    print("label stat")
    s = pd.Series(stat_label)
    print(s.describe())
    print("length after :",len(df["B"]))



