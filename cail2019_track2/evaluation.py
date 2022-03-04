#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:huanghui
# datetime:2019/9/30 10:28
import json
from utils.evaluate import evaluate
from utils.predict import BERTModel
import logging
import numpy as np
import re

logging.basicConfig(level=logging.INFO)

def readThreshold(fname):
    """读取json阈值文件"""
    with open(fname, "r", encoding="utf-8") as f:
        dic = json.loads(f.readline())
    return dic

def add_(arr1, arr2):
    for i in range(len(arr1)):
        arr1[i].extend(arr2[i])
        arr1[i] = list(set(arr1[i]))
    return arr1

def re_match(text, feature):
    pred = np.zeros(20, dtype=np.int32)
    if feature is None:
        return pred
    for i in feature.keys():
        pred[i-1] = any([re.match(key, text) is not None for key in feature[i]])
    return pred

def getMatch(feature, sentences, model):
    """添加规则测试，采用正则表达式匹配，最后与模型结果取并集"""
    re_pred = np.zeros((len(sentences), 20), dtype=np.int32)
    for idx, sent in enumerate(sentences):
        re_pred[idx] = re_match(str(sent), feature)
    re_pre = model.rematch(re_pred)
    return re_pre

def load_file(filename):
    f = open(filename, "r", encoding='utf-8')
    all_sentence = []
    all_label = []
    for line in f:
        pre_doc = json.loads(line)
        for sent in pre_doc:
            all_sentence.append(sent["sentence"])
            all_label.append(sent["labels"])
    f.close()
    return all_sentence, all_label

if __name__ == '__main__':

    task = "divorce"

    ##这里传入切分好的测试数据，这里由于是整理代码做测试，随便导入训练数据集测试下
    sentences, labels = load_file("data/divorce/train_selected.json")

    logging.info("开始载入bert模型")
    model_1 = BERTModel(task=task, pb_model="pb/divorce/model.pb",
                        tagDir="data/divorce/tags.txt", threshold=[0.5] * 20,
                        vocab_file="/userhome/30/hjpan/legal/pretrained/chinese_L-12_H-768_A-12/vocab.txt")

    logging.info("bert模型载入完毕，开始进行预测！！！\n")
    logging.info("模型开始预测\n")
    predicts_1 = model_1.getAllResult(sentences)
    logging.info("模型预测结束\n")

    logging.info("模型每个类别f值计算如下：\n")
    score_1, f1_1 = evaluate(predict_labels=predicts_1, target_labels=labels, tag_dir="data/divorce/tags.txt")
    logging.info(f1_1)
    logging.info("总评分如下： {}".format(score_1))

# chinese bert
# INFO:root:{'1': 0.96, '2': 0.92, '3': 0.92, '4': 0.94, '5': 0.92, '6': 0.94, '7': 0.92, '8': 0.97, '9': 0.98, '10': 0.87, '11': 0.84, '12': 0.15, '13': 0.85, '14': 0.53, '15': 0.83, '16': 0.83, '17': 0.75, '18': 0.15, '19': 0.32, '20': 0.64}
# INFO:root:总评分如下： 0.8354149736395091














