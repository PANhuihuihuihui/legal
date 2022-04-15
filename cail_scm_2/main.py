import json
import logging
import sys
import time
import os

import torch

from model import BertSimMatchModel

logging.disable(sys.maxsize)

start_time = time.time()
MODEL_DIR = "/home/huijie/legal/cail_scm_1/ms_scm"
input_path = "/home/huijie/legal/cail_scm_2/data/test/input.txt"
if os.path.exists(MODEL_DIR+"/output"):
    pass
else:
    os.mkdir(MODEL_DIR+"/output")
output_path = MODEL_DIR+ "/output.txt"

if len(sys.argv) == 3:
    input_path = sys.argv[1]
    output_path = sys.argv[2]

inf = open(input_path, "r", encoding="utf-8")
ouf = open(output_path, "a", encoding="utf-8")


model = BertSimMatchModel.load(MODEL_DIR, torch.device("cuda:0"))

text_tuple_list = []
for line in inf:
    line = line.strip()
    items = json.loads(line)
    a = items["A"]
    b = items["B"]
    c = items["C"]
    text_tuple_list.append((a, b, c))

results = model.predict(text_tuple_list)

for label, _ in results:
    # print(str(label), _)
    print(str(label), file=ouf)

inf.close()
ouf.close()

end_time = time.time()
spent = end_time - start_time
print("numbers of samples: %d" % len(results))
print("time spent: %.2f seconds" % spent)
