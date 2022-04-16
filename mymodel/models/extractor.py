from types import NoneType
import torch
from torch import nn
import torch.nn.functional as F

from transformers import AutoModel



class CNN_Text(nn.Module):
    def __init__(self,embed_dim,sent_length,dropout = 0.1):
        super(CNN_Text, self).__init__()
        Ci = 1
        Co = embed_dim*2
        K = sent_length*+1 #kernel_sizes
        self.conv = nn.Conv2d(Ci, Co, (K, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):    
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = F.relu(self.conv(x)).squeeze(3)   #(N, Co, W)

        x = F.max_pool1d(x, x.size(2)).squeeze(2) # (N, Co)
        x = self.dropout(x)  # (N, Co)
        return x

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
        self.classifier = nn.Linear(self.hidden_size, num_labels)
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
    def __init__(self,extractor):
        super(Fuser,self).__init__()
        self.extractor = extractor
        self.cnn = CNN_Text()
        embed_size = extractor.back_bone.config.hidden_size
        self.lstm = nn.LSTM(embed_size,embed_size//2,batch_first = True,bidirectional=True)
    def forward(self,x):
        x= x.permuate(1,0,2,3)
        Mc = []
        Mk = []
        for sent in range(x.shape[0]):
            ec,ek = self.extractor(x[sent,:])
            Mc.append(ec)
            Mk.append(ek) # b S E
        Mc = torch.stack(Mc,dim = 1).squeeze(2) # b d*S E
        Mk = torch.stack(Mk,dim = 1).squeeze(2) # b d*S E
        Mc = self.cnn(Mc) # b Co
        Mk,_ = self.lstm(Mk) #b ds E
        Mk = Mk.sum(dim = 1) # b E
        x = torch.cat((Mc,Mk),1) # b (Co + E)
        return x 














        
        
        for 
        x = 
        

def model_build(args):
    if args.stage ==1:
        model = Extractor()
    else:
        model = None
    return model