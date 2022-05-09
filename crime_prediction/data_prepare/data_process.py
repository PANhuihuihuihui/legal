import string
import re
import json
from tqdm import tqdm
from pathlib import Path
import os
import codecs


codecs_out = lambda x : codecs.open(x, 'w', 'utf-8')
codecs_in = lambda x : codecs.open(x, 'r', 'utf-8')
json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
json_dumps = lambda d: json.dumps(d, indent=2, ensure_ascii=False)

def ifCrimeIn(fileText):
    """doesn't matter"""
    pass
def writeFile(tmp):
    with codecs.open("/home/huijie/legal/crime_prediction/data_prepare/crime_prediction.json", 'a+') as json_file:
        for instant in tmp:
            json.dump(instant, json_file,ensure_ascii=False)
            json_file.write('\n')
def cleanTextContent(example):
    tmp = re.sub(r'^(\d+).', '', example)
    tmp = re.sub(r'<mark\d+>', '', tmp)
    tmp = re.sub(r'^\((ix|iv|v?i{0,3})\)', '', tmp)
    tmp = re.sub(r'^\((\d+)\)', '', tmp)
    new_str = tmp.translate(str.maketrans('','',string.punctuation)).replace("’s",'').replace("s’",'s')
    new_str = new_str.replace("「",'').replace("」",'').replace("  "," ")
    return new_str
def cleanTextBegin(example):
    tmp = re.sub(r'^(\d+).', '', example)
    tmp = re.sub(r'<mark\d+>', '', tmp)
    tmp = re.sub(r'^\((ix|iv|v?i{0,3})\)', '', tmp)
    tmp = re.sub(r'^\((\d+)\)', '', tmp)
    tmp = re.sub(r'（.*?）', '', tmp)
    new_str = tmp.translate(str.maketrans('','',string.punctuation)).replace("’s",'').replace("s’",'s')
    new_str = new_str.replace("「",'').replace("」",'').replace(" ","")
    return new_str

def data_generate(ch_file):
    labels = ['盜竊罪', '爆竊罪', '猥褻侵犯罪', '侵害人身罪', '不小心駕駛罪', '販運危險藥物罪', '入屋犯法罪', 
            '搶劫罪', '以欺騙手段取得財產罪', '販毒罪', '販運毒品罪', '普通襲擊罪', '洗黑錢罪', '串謀詐騙罪', 
            '處理贓物罪', '刑事恐嚇罪', '串謀勒索罪', '非法入境罪', '強姦罪', '非禮罪', '縱火罪', '扒竊罪', 
            '管有危險藥物罪', '危險駕駛引致他人死亡罪', '襲擊致造成身體傷害罪', '管有虛假文書罪', '使用虛假文書罪',
             '危險駕駛引致他人身體受嚴重傷害罪', '危險駕駛罪']
    zh_topics = []
    for crime in labels:
        zh_topics.append(crime)
        zh_topics.append('“' + crime[:-1] + '”' + crime[-1])
    labels = zh_topics
    tmp = []
    data = json_load(ch_file)
    beginTextList = []
    contentList = []
    for properity, content in data['judgement']:
        if properity == ["para"]:
            beginTextList.append(cleanTextBegin(content))
        contentList.append(cleanTextContent(content))
    txt = ' '.join(contentList)
    beginText = ' '.join(beginTextList)[:256]
    #对于每个label（有3000多个同文件含有多个罪名情况）
    for label in labels:
        #每个文件同label可能出现多次，对于每次出现
        for m in re.finditer(label, txt):
            #确定起始位置
            s=m.span()[0]
            e=m.span()[1]
            #如果前后够长则前取250，后取250，否则取到头尾端
            s=s-128
            if s<0:
                s=0
            e=e+128
            if e>=len(txt)-1:
                e=len(txt)-1
            #对于每段文本单独处理成一个项,结束后总共32498个项目
            temp_dict=dict()
            temp_dict['label']=label.replace("“",'').replace("”",'')
            temp_dict['begin'] = beginText
            temp_dict['text']=txt[s:e].replace(label,'罪')


            tmp.append(temp_dict)
    return tmp

    
if __name__ == '__main__':
    paths = [] 
    for path in Path('/home/xijia/nlp/log3').rglob('*.json'):
        paths.append(path)
    print(len(paths))
    tmp = []
    for path in tqdm(paths):
        tmp.extend(data_generate(path))
    print(tmp[:5])
    print(len(tmp))
    writeFile(tmp)
