#coding=utf-8
#-*- coding: UTF-8 -*-
import sys
'''python3'''

import os
import re
import json
import time
import codecs
import random
import argparse
import functools
import collections
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords
from string import punctuation
import jieba.posseg as pseg

codecs_out = lambda x : codecs.open(x, 'w', 'utf-8')
codecs_in = lambda x : codecs.open(x, 'r', 'utf-8')

json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
json_dumps = lambda d: json.dumps(d, indent=2, ensure_ascii=False)
def json_dumpl(d, p) :
    output = codecs_out(p)
    for dt in d :
        output.write(json.dumps(dt, ensure_ascii=False)+'\n')
    output.flush()
    output.close()
def json_loadl(p) :
    lines = codecs_in(p).readlines()
    to_ret = list(map(json.loads, lines))
    return to_ret

""" ===== ===== ===== ===== ===== ===== """
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from itertools import permutations

def preprocess(text, isLabel=False):
    #小写化
    text = text.lower()
    #去除特殊标点
    for c in punctuation:
        text = text.replace(c, ' ')
    #分词
    wordLst = nltk.word_tokenize(text)
    #去除停用词
    filtered = [w for w in wordLst if w not in stopwords.words('english')]
    #仅保留名词或特定POS   
    refiltered =nltk.pos_tag(filtered)
    filtered = [w for w, pos in refiltered if pos.startswith(('NN', 'JJ', 'VB'))]
    #词干化
    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]

    # if isLabel:
    #     perms = list(permutations(filtered))
    #     return [' '.join(p) for p in perms]
    return " ".join(filtered)

# log3 = ch, log2 = en
def get_file_paths(input_dir='log2') :
    to_ret = []
    dirs = [t for t in os.listdir(input_dir) if not t[0]=='.']
    for dir1 in dirs :
        pt = os.path.join(input_dir, dir1, 'data')
        files = [t for t in os.listdir(pt) if not t[0] == '.']
        to_ret+= [os.path.join(pt, file) for file in files]
    return to_ret

def get_zm(input_file) :
    data = json_load(input_file)['judgement']
    contents = []
    for t in data :
        re_result = re.match(r'.*(裁定.*?罪).*', t[1])
        if re_result is None :
            continue
        nt = re_result.group(1)
        if len(nt) < 10 :
            continue
        contents.append(nt)
        # print(re_result.group(1))
    # exit()
    return contents


def visit_with_trie(input_files) :
    trie = collections.Counter()

    cnt = 0
    tot_cnt = 0

    crimes = ['trafficking in a dangerous drug', 'obstruction', 'causing death by dangerous driving', 'possession of a false instrument', 'handling stolen goods', 
            'inflicting grievous bodily harm', 'illegal entry', 'causing grievous bodily harm by dangerous driving', 'using a false instrument', 'possessing a false instrument', 
                'rape', 'robbery', 'criminal intimidation', 'arson', 'fraud', 'assault occasioning actual bodily harm', 'possessing a dangerous drug', 'dangerous driving', 'burglary', 
                'pickpocketing', 'careless driving', 'murder', 'possession of a dangerous drug', 'conspriacy to defraud', 'money laundering', 'theft', 'common assault', 'indecent assault']
    
    topics = []
    for crime in crimes:
        topics.append(preprocess(crime, True))

    def visit_t(file) :
        data = json_load(file)['judgement']
        noted = [0] * len(crimes)
        for line in data :
            line = preprocess(line[1])
            for i, crime in enumerate(crimes):
                if crime in line and not noted[i]:
                    noted[i] = 1
        trie[sum(noted)] += 1

    for file in tqdm(input_files) :
        visit_t(file)
    
    dictt = [t for t in list(trie.most_common())]
    print(dictt)
#     topics_cnt = collections.Counter()
#     extracted = []
#     not_extracted = []
#     for word, cnt in dictt:
#         flag = False
#         for topic in topics:
#             if topic in word:
#                 topics_cnt[topic] += cnt
#                 flag = True
#         if flag:
#             continue
#         # if sum([c in word[:-1] for c in inc]) != 0:
#         #     extracted.append((word, cnt))
#         #     continue
#         words = pseg.cut(word)
#         for _, flag in words:
#             if flag in flags and sum([c in word[:-1] for c in not_inc]) == 0:
#                 # print(word, flag)
#                 extracted.append((word, cnt))
#                 break
#         else:
#             not_extracted.append((word, cnt))

#     result = []
#     for i, (w1, c1) in enumerate(extracted):
#         for w2, c2 in extracted[i+1:]:
#             a, b = (w1, w2) if len(w1) < len(w2) else (w2, w1)
#             if a in b:
#                 break
#         else:
#             result.append((w1, c1 + c2))
#     # extracted = ['盜竊罪', '販運危險藥物罪', '入屋犯法罪', '搶劫罪', '以欺騙手段取得財產罪']
#     # dictt = [t for t in dictt if 3937>t[1]>=10]
#     # dictt = [t for t in dictt if not any(t[0][-len(w):] == w for w in extracted)]
#     print('提取出的主题:', len(topics_cnt))
#     print(topics_cnt)
#     print('（除以上主题外）可能的罪名:', len(result))
#     print(result[:30])
#     print('不太可能的罪名:', len(not_extracted))
#     print(not_extracted[:30])

if __name__ == '__main__':
    files = get_file_paths()
    visit_with_trie(files)
    
    # for file in tqdm(files) :
    #     # print(file)
    #     # if not 'HKCA/data/2019_207' in file :
    #     #     continue
    #     contents = get_zm(file)
    #     if len(contents) > 0 :
    #         print(file, contents)
