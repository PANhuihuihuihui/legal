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
from tqdm import tqdm
from pathlib import Path
from nltk.corpus import stopwords
from string import punctuation
from functools import lru_cache
from threading import Thread

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

def findCrime(ch_file):
    ch_file = json_load(ch_file)
    txt = ' '.join([p for _, p in ch_file['judgement']])
    x = re.search(r'裁定.*(‘|“.+?’|”罪)', txt)
    if x != None:
        print(x)
        input()

if __name__ == '__main__':
    paths = [] 
    for path in Path('log3').rglob('*.json'):
        paths.append(path)
    print(len(paths))
    for path in tqdm(paths):
        findCrime(path)