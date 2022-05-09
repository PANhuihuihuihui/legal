#coding=utf-8
#-*- coding: UTF-8 -*-
import sys
'''python3'''

import os
import re
import json
import codecs
import collections
from tqdm import tqdm
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
crimes = ['盜竊罪', '猥褻侵犯罪', '侵害人身罪', '不小心駕駛罪', '販運危險藥物罪', '入屋犯法罪', '搶劫罪', '以欺騙手段取得財產罪', '販毒罪', '販運毒品罪', '普通襲擊罪', 
        '洗黑錢罪', '串謀詐騙罪', '處理贓物罪', '刑事恐嚇罪', '串謀勒索罪', '非法入境罪', '強姦罪', '非禮罪', '縱火罪', '扒竊罪']

# log3 = ch, log2 = en
def get_file_paths(input_dir='/home/xijia/nlp/log3') :
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

    topics = []
    for crime in crimes:
        topics.append(crime)
        topics.append('「' + crime[:-1] + '」' + crime[-1])
        topics.append('“' + crime[:-1] + '”' + crime[-1])

    def visit_t(file) :
        data = json_load(file)['judgement']
        result = []
        for line in data :
            line = line[1]
            for i, topic in enumerate(topics):
                index = line.find(topic)
                if index != -1:
                    trie[crimes[i // 3]] += 1
                    result.append(line[max(0, index - 64): min(len(line), index + 64)])
        return result
    
    result = []
    for file in tqdm(input_files) :
        result.append(visit_t(file))
    
    dictt = [t for t in list(trie.most_common())]
    print(dictt)
    for r in result[:20]:
        for t in r:
            print(t)
    
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
