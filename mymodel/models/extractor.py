from types import NoneType
import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel



class CNN_Text(nn.Module):
    
    def __init__(self, args,embed_dim,sent_length):
        super(CNN_Text, self).__init__()
        self.args = args
    
        D = args.embed_dim
        Ci = 1
        Co = args.kernel_num
        K = sent_length*+1 #kernel_sizes
        self.convs = nn.Conv2d(Ci, Co, (K, embed_dim))
        self.dropout = nn.Dropout(args.dropout)

        if self.args.static:
            self.embed.weight.requires_grad = False

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
    
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Extractor(nn.Module):
    """ Extract the infomation inside a document """
    def __init__(self,back_bone_name = "bert-base-multilingual-uncased",num_labels = 60,stage = 1):
        super(Extractor,self).__init__()
        self.back_bone = AutoModel.from_pretrained(back_bone_name)
        self.hidden_size = self.back_bone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.stage = stage
    def forward(self,x):
        ec = self.back_bone(x).pooler_output
        ec = self.dropout(ec)
        ek = F.sigmoid(self.classifier(ec))
        return ec,ek


class Fuser(nn.Module):
    """
     build Mc Mk and form a document representation d
     inputs:[b* D * S * E]

    
    """
    def __init__(self,extractor) -> None:
        super().__init__()
        self.extractor = extractor
        self.cnn = CNN_Text()
        self.lstm = 
    def forward(x):
        x= x.permuate(1,0,2,3)
        Mc = []
        Mk = []
        for sent in range(x.shape[0]):
            ec,ek = self.extractor(x[sent,:])
            Mc.append(ec)
            Mk.append(ek) # b S E
        Mc = torch.stack(Mc,dim = 1).squeeze(2) # b d*S E
        Mc = self.cnn(Mc)
        Mk = torch.stack(Mk,dim = 1)




        
        
        for 
        x = 
        

def model_build(args):
    if args.stage ==1:
        model = Extractor()
    else:
        model = None
    return model