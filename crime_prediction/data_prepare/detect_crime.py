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
flags = ['n', 'f', 's', 'nr', 'nrt', 'ns', 'nt', 'nw', 'nz', 'an', 'v', 'i']
topics = ['藥物', '恐嚇', '駕駛', '襲擊', '勒索', '猥褻', '侵犯', '強姦', '縱火', '損壞', '入境', '非禮', '侵犯', '財產', '詐騙', '毒', '盜竊', '贓物', '爆竊', '人身', '串謀', '入屋', '黑錢', '扒竊', '青少年']
not_inc = ['的', '項', '該', '事', '訴', '認', '脫', '服定', '控', '無', '告', '一', '件', '之', '此', '等', '這', '其', '些', '案', '個', '種', '條', '何', '被']

# log3 = ch, log2 = en
def get_file_paths(input_dir='log3') :
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
        if len(nt) < 5 :
            continue
        contents.append(nt)
        # print(re_result.group(1))
    # exit()
    return contents


def visit_with_trie(input_files) :
    trie = collections.Counter()

    cnt = 0
    tot_cnt = 0

    def visit_t(file) :
        nonlocal cnt, tot_cnt

        data = json_load(file)['judgement']
        tmp = []
        for line in data :
            line = line[1]
            p_zui = [i for i, c in enumerate(line) if c == '罪']
            for i in p_zui :
                for j in range(1, min(10, i+1)) :
                    word = line[i-j:i+1]
                    trie[word] += 1
                    tmp.append(word)
        
        if '無罪' in tmp:
            # print(tmp)
            tot_cnt += 1
            results = []
            resultss = []
            for _, t in data:
                result = re.match(r'.*(裁定.*?罪).*', t)
                if result != None:
                    results.append(result.group(1))
                result = t.find('無罪')
                if result != -1:
                    resultss.append(t[result-10:result+10])
            if results != []:
                cnt += 1
            else:
                if random.random() < 0.05:
                    print(resultss)
                    print(tmp)
                    input()
        
    for file in tqdm(input_files) :
        visit_t(file)
        
    print(cnt, tot_cnt, cnt / tot_cnt)
    input()
    dictt = [t for t in list(trie.most_common()) if t[1]>=100]

    
    topics_cnt = collections.Counter()
    extracted = []
    not_extracted = []
    for word, cnt in dictt:
        flag = False
        for topic in topics:
            if topic in word:
                topics_cnt[topic] += cnt
                flag = True
        if flag:
            continue
        # if sum([c in word[:-1] for c in inc]) != 0:
        #     extracted.append((word, cnt))
        #     continue
        words = pseg.cut(word)
        for _, flag in words:
            if flag in flags and sum([c in word[:-1] for c in not_inc]) == 0:
                # print(word, flag)
                extracted.append((word, cnt))
                break
        else:
            not_extracted.append((word, cnt))

    result = []
    for i, (w1, c1) in enumerate(extracted):
        for w2, c2 in extracted[i+1:]:
            a, b = (w1, w2) if len(w1) < len(w2) else (w2, w1)
            if a in b:
                break
        else:
            result.append((w1, c1 + c2))
    # extracted = ['盜竊罪', '販運危險藥物罪', '入屋犯法罪', '搶劫罪', '以欺騙手段取得財產罪']
    # dictt = [t for t in dictt if 3937>t[1]>=10]
    # dictt = [t for t in dictt if not any(t[0][-len(w):] == w for w in extracted)]
    print('提取出的主题:', len(topics_cnt))
    print(topics_cnt)
    print('（除以上主题外）可能的罪名:', len(result))
    print(result[:30])
    print('不太可能的罪名:', len(not_extracted))
    print(not_extracted[:30])

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
