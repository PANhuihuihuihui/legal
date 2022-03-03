#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:huanghui
# datetime:2019/9/30 10:28

class Judger:
    # Initialize Judger, with the path of tag list
    def __init__(self, tag_path):
        self.tag_dic = {}
        f = open(tag_path, "r", encoding='utf-8')
        self.task_cnt = 0
        for line in f:
            self.task_cnt += 1
            self.tag_dic[line.strip()] = self.task_cnt


    # Format the result generated by the Predictor class
    @staticmethod
    def format_result(result):
        rex = {"tags": []}
        res_art = []
        for x in result["tags"]:
            if not (x is None):
                res_art.append(int(x))
        rex["tags"] = res_art

        return rex

    # Gen new results according to the truth and users output
    def gen_new_result(self, result, truth, label):

        s1 = set()
        for tag in label:
            s1.add(self.tag_dic[tag.replace(' ', '')])
        s2 = set()
        for name in truth:

            s2.add(self.tag_dic[name.replace(' ', '')])

        for a in range(0, self.task_cnt):
            in1 = (a + 1) in s1
            in2 = (a + 1) in s2
            if in1:
                if in2:
                    result[0][a]["TP"] += 1
                else:
                    result[0][a]["FP"] += 1
            else:
                if in2:
                    result[0][a]["FN"] += 1
                else:
                    result[0][a]["TN"] += 1

        return result

    # Calculate precision, recall and f1 value
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    @staticmethod
    def get_value(res):
        if res["TP"] == 0:
            if res["FP"] == 0 and res["FN"] == 0:
                precision = 1.0
                recall = 1.0
                f1 = 1.0
            else:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
        else:
            precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
            recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
            f1 = 2 * precision * recall / (precision + recall)

        return precision, recall, f1

    # Generate score
    def gen_score(self, arr):
        sumf = 0
        f1 = {}
        y = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        i = 0
        for x in arr[0]:
            i += 1
            p, r, f = self.get_value(x)
            f1[str(i)] = round(f, 2)
            sumf += f
            for z in x.keys():
                y[z] += x[z]

        _, __, f_ = self.get_value(y)



        return (f_ + sumf * 1.0 / len(arr[0])) / 2.0, f1

    # Test with ground truth path and the user's output path
    def test(self, truth_label, pre_label):
        cnt = 0
        result = [[]]
        for a in range(0, self.task_cnt):
            result[0].append({"TP": 0, "FP": 0, "TN": 0, "FN": 0})

        for i in range(len(truth_label)):
            cnt += 1
            result = self.gen_new_result(result, truth_label[i], pre_label[i])
        return result

def evaluate(predict_labels, target_labels, tag_dir):
    """传入预测标签，目标标签，tags地址。"""
    """标签需要真实标签，而不是id，例如["dv1","dv2"]"""
    judger = Judger(tag_dir)
    reslt = judger.test(target_labels, predict_labels)

    score, f1 = judger.gen_score(reslt)
    return score, f1