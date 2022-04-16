# why so many loop?
# file pattern [ { sentence: "", label: ""}]

# load file
import json
import codecs
with codecs.open("/home/huijie/legal/mymodel/data/legal_element.json", 'a+') as json_file:
    with open('/home/xiangwei/classify/divorce.json','r',encoding='utf8')as fp:
        lines=fp.readlines()
        for line in lines:
            list_data = json.loads(line)
            for data in list_data:
                temp=[-1]*20
                for label in data['labels']:
                    index = int(label[2:])-1
                    temp[index]=1
                data['labels'] = temp + [-1]*40
                data['class'] = "divorce"
                json.dump(data, json_file,ensure_ascii=False)
                json_file.write('\n')
               
    with open('/home/xiangwei/classify/labor.json','r',encoding='utf8')as fp:
        lines=fp.readlines()
        for line in lines:
            list_data = json.loads(line)
            for data in list_data:
                temp=[-1]*20
                for label in data['labels']:
                    index = int(label[2:])-1
                    temp[index]=1
                data['labels'] = [-1]*20 + temp + [-1]*20
                data['class'] = "labor"
                json.dump(data, json_file,ensure_ascii=False)
                json_file.write('\n')
   
    with open('/home/xiangwei/classify/loan.json','r',encoding='utf8')as fp:
        lines=fp.readlines()
        for line in lines:
            list_data = json.loads(line)
            for data in list_data:
                temp=[-1]*20
                for label in data['labels']:
                    index = int(label[2:])-1
                    temp[index]=1
                data['labels'] = [-1]*40 + temp
                data['class'] = "loan"
                json.dump(data, json_file,ensure_ascii=False)
                json_file.write('\n')


