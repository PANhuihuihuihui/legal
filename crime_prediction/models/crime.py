import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel


class MultiClassify(nn.Module):
    def __init__(self,args):
        super(MultiClassify, self).__init__()
        self.back_bone = AutoModel.from_pretrained(args.back_bone)
        self.hidden_size = self.back_bone.config.hidden_size
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax=nn.Softmax(dim=-1)
        self.classifier = torch.nn.Linear(self.hidden_size, args.num_labels)

    def forward(self,x):
        x = self.back_bone(**x).pooler_output
        x = self.dropout(x)
        x = self.classifier(x)
        x = self.softmax(x)
        return x




def build(args):
    model = MultiClassify(args)
    return model
