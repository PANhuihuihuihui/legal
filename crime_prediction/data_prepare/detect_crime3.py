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

def preprocess(text):
    #小写化
    text = text.lower()
    #去除特殊标点
    # for c in punctuation:
    #     text = text.replace(c, ' ')
    filtered = text.split()
    # filtered = [w for w in text if w not in stopwords.words('english')]
    return filtered

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
    kw = ['offence', 'count', 'counts']
    skip = ['in', 'for', 'to']
    crimes = ['盜竊罪', '爆竊罪', '猥褻侵犯罪', '侵害人身罪', '不小心駕駛罪', '販運危險藥物罪', '入屋犯法罪', '搶劫罪', '以欺騙手段取得財產罪', '販毒罪', '販運毒品罪', '普通襲擊罪', '洗黑錢罪', '串謀詐騙罪', 
            '處理贓物罪', '刑事恐嚇罪', '串謀勒索罪', '非法入境罪', '強姦罪', '非禮罪', '縱火罪', '扒竊罪', '管有危險藥物罪', '危險駕駛引致他人死亡罪', '襲擊致造成身體傷害罪', '管有虛假文書罪', '使用虛假文書罪',
             '危險駕駛引致他人身體受嚴重傷害罪', '危險駕駛罪']
    zh_topics = []
    for crime in crimes:
        zh_topics.append(crime)
        zh_topics.append('「' + crime[:-1] + '」' + crime[-1])
        zh_topics.append('“' + crime[:-1] + '”' + crime[-1])
    zh_en = [collections.Counter() for _ in crimes]

    def visit_t(file) :
        try:
            file_ = file.replace('log2', 'log3')
            data_ = json_load(file_)['judgement']
        except:
            return

        data = json_load(file)['judgement']
        en_topics = []
        for line in data :
            line = preprocess(line[1])
            # # 定位词 + of
            p_zui = [i + 2 for i, c in enumerate(line) if \
                (i != len(line) -1) and (c in kw) and (line[i + 1] == 'of')]
            for i in p_zui :
                for j in range(1, min(len(line) - i, 6)):
                    word = ' '.join(line[i:i+j])
                    trie[word] += 1
                    en_topics.append(word)
        
        for line in data_:
            for i, topic in enumerate(zh_topics):
                if topic in line[1]:
                    for t in en_topics:
                        zh_en[i // 3][t] += 1
        
    for file in tqdm(input_files) :
        visit_t(file)
    
    zh_en = [[t for t in list(d.most_common(20))] for d in zh_en]
    for i, crime in enumerate(crimes):
        print(crime)
        print(zh_en[i])
    dictt = [t for t in list(trie.most_common()) if t[1]>=100]
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
