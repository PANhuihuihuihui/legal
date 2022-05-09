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
import collections
import numpy as np
# import pandas as pd
from tqdm import tqdm
from pathlib import Path
# from bs4 import BeautifulSoup
# import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from string import punctuation
# from translate import translate
from functools import lru_cache
from threading import Thread
from multiprocessing import Process

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
def mkdir_p(path):
    try:
        os.makedirs(path)
    except:
        pass

embed_dict = {}
stop_words = stopwords.words('english')
concat = lambda x: [a for s in x for a in s]
concat_types = lambda text, idxs: [label for idx in list(idxs) for label in text['judgement'][idx][0]]
concat_contents = lambda text, idxs: '\n'.join([text['judgement'][idx][1] for idx in list(idxs)])

@lru_cache
def cos(a, b):
    embed_a = embed_dict.get(a, embed_dict['unk'])
    embed_b = embed_dict.get(b, embed_dict['unk'])
    norm_a, norm_b = np.linalg.norm(embed_a), np.linalg.norm(embed_b)
    return np.dot(embed_a, embed_b) / (norm_a * norm_b)

def preprocess(text):
    #小写化
    text = text.lower()
    #去除特殊标点
    for c in punctuation:
        text = text.replace(c, ' ')
    text = text.split()
    filtered = [w for w in text if w not in stopwords.words('english')]
    return filtered

process_chi = lambda x: [item[1] for item in x['judgement'] if len(item[1].strip())]
process_eng = lambda x: [preprocess(item[1]) for item in x['judgement'] if len(item[1].strip())]
process_eng_ = lambda x: [item[1] for item in x['judgement'] if len(item[1].strip())]
process_c2e = lambda x: [preprocess(item) for item in x['judgement'] if len(item.strip())]
get_index = lambda x, i: x[i][0].split('.')[0]

def load(path='/home/xijia/nlp/glove/glove.6B.50d.txt'):
    print('loading glove')
    embed_dict = collections.defaultdict(lambda:np.random.random([300]))
    # with open(path) as f:
    #     for line in f:
    #         line = line.split()
    #         embed_dict[line[0]] = np.asarray(line[1:], 'float32')
    print('glove loaded')
    return embed_dict
    
def align(c2epath, embed_dict):
    epath = eng_log + str(c2epath)[len(eng_log):]
    cpath = chi_log + str(c2epath)[len(eng_log):]
    c2e, eng, chi = json_load(c2epath), json_load(epath), json_load(cpath)

    A, A_original, B, B_ = process_c2e(c2e), process_chi(chi), process_eng(eng), process_eng_(eng)
    # print(A, B)

    """
    !!!! 解决 A 与 A_original 不等长的问题
    """
    if len(A) != len(A_original):
        # split paragraph by \n
        num_newlines = 0
        tot_newlines = 0
        A_ = []
        index = 0
        for p in A_original:
            num_newlines = len([s for s in p.split('\n') if len(s.strip())])
            tot_newlines += num_newlines
            p = concat(A[index : index + num_newlines])
            index = index + num_newlines
            A_.append(p)
        if len(A) == tot_newlines:
            A = A_
        else:   
            print('cannot align chinese text and its translation, aborting')
            sys.exit()

    assert len(A) == len(A_original)
    m, n = len(A), len(B)
    # print('m, n', m, n)
    # print(c2epath, len(A), len(B))
    
    # A, B, A_original, m, n 直接从上一层的全局变量中获得。

    @lru_cache
    def sim(l1p1, l1p2, l2p1, l2p2):
        # print(l1p1, l1p2, l2p1, l2p2)
        l1 = concat(A[l1p1:l1p2])
        l2 = concat(B[l2p1:l2p2])
        # print(l1, l2)
        if len(l1) == 0 or len(l2) == 0:
            return 0
        s1, s2 = 0, 0
        for word in l1:
            s1 += max([cos(word, wordd) for wordd in l2])
        s1 /= len(l1)
        for word in l2:
            s2 += max([cos(word, wordd) for wordd in l1])
        s2 /= len(l2)
        # print(s1 + s2)
        return s1 + s2

    @lru_cache
    def dp(px=0, py=0):
        # print(px, py)
        # 通过序号剪枝
        try:
            va = int(A_original[px][:5].split('.')[0])
        except:
            va = None
        try:
            vb = int(B[py][0].split('.')[0])
        except:
            vb = None

        if (va != None and vb != None) and va != vb:
            return float('-inf'), []

        if abs(px-py) > max(2 * abs(m-n), 5):
            return float('-inf'), []
        if px == m - 1:
            return sim(px, px+1, py, n), [[[px], range(py, n)]]
        if py == n - 1:
            return sim(px, m, py, py+1), [[range(px, m), [py]]]
        
        to_ret = float('-inf')
        to_ret_list = None
        for npy in range(py + 1, min(n, py+5)): # 一个基于先验的剪枝：最多允许4句话一组
            sim_t, listt = dp(px + 1, npy)
            sim_t += sim(px, px+1, py, npy) * (1 + npy - py) #
            if sim_t > to_ret:
                to_ret = sim_t
                to_ret_list = listt + [[[px], range(py, npy)]]
        for npx in range(px + 1, min(m, px+5)): # 一个基于先验的剪枝：最多允许4句话一组
            sim_t, listt = dp(npx, py + 1)
            sim_t += sim(px, npx, py, py+1) * (1 + npx - px)
            if sim_t > to_ret:
                to_ret = sim_t
                to_ret_list = listt + [[range(px, npx), [py]]]
        return to_ret, to_ret_list

    score, contents = dp()
    if contents == None:
        print(c2epath, 'failed')
        return

    to_ret = {}
    contents = contents[::-1]
    keys = ['en_types', 'ch_types', 'en_contents', 'ch_contents']
    dj = to_ret['judgement'] = [{k:[] for k in keys} for _ in range(len(contents))]
    for i, (ch_idx, en_idx) in enumerate(contents):
        dj[i]['en_types'] = concat_types(eng, en_idx)
        dj[i]['ch_types'] = concat_types(chi, ch_idx)
        dj[i]['en_contents'] = concat_contents(eng, en_idx)
        dj[i]['ch_contents'] = concat_contents(chi, ch_idx)
   
    for key in chi:
        if key != 'judgement':
            to_ret['ch_' + key] = chi[key]
            to_ret['en_' + key] = eng[key]

    path = save_path + str(c2epath)[len(eng_log):]
    mkdir_p('/'.join(path.split('/')[:-1]))
    json_dump(to_ret, path)
    print(path, 'done')


if __name__ == '__main__':
    eng_log = 'log2/'
    chi_log = 'log3/'
    c2e_log = 'log3t/'
    save_path = 'logp2/'
    
    paths = [] 
    for path in Path(c2e_log).rglob('*.json'):
        if not os.path.exists(save_path + str(path)[len(c2e_log):]):
            paths.append(path)
    num_paths = len(paths)
    
    embed_dict = load()

    # for path in tqdm(paths):
    #     start = time.time()
    #     align(path, embed_dict)
    #     end = time.time()
    #     print(f'finished in {end - start:.2f}')

    threads = [Process(target=align, args=(c2epath, embed_dict)) for c2epath in paths]
    max_threads = 8
    failed = []
    for i in tqdm(range(num_paths // max_threads + 1)):
        num_threads = min(max_threads, num_paths - i * max_threads)
        for j in range(num_threads):
            threads[i * max_threads + j].start()
        for j in range(num_threads):
            p = i * max_threads + j
            if j == 0:
                threads[p].join(6000)
            else:
                threads[p].join(10)
            if threads[p].is_alive():
                threads[p].terminate()
                failed.append(p)
                print(failed, 'timeout')
        print(failed)