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

class Extractor(nn.Module):
    """ Extract the infomation inside a document """
    def __init__(self,args,back_bone):
        super(Extractor,self).__init__()
        self.stage = args.stage
        self.back_bone = back_bone
        self.hidden_size = self.back_bone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, args.num_labels)

        
    def forward(self,x):
        ec = self.back_bone(x).pooler_output
        ec = self.dropout(ec)
        ek = self.classifier(ec)
        return ec,ek if self.stage == 1 else ec,F.sigmoid(ek)
    #  def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)


class Fuser(nn.Module):
    """
     build Mc Mk and form a document representation d
     inputs:[b* D * S * E]

    """
    def __init__(self,args,extractor):
        super(Fuser,self).__init__()
        self.extractor = extractor
        self.cnn = CNN_Text()
        embed_size = args.hidden_size
        self.lstm = nn.LSTM(embed_size,embed_size//2,batch_first = True,bidirectional=True)
    def forward(self,x):
        """
        inputs: b d s e
        """
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

def build_extractor(args):
    back_bone = AutoModel.from_pretrained(args.back_bone_name)
    model = Extractor(args,back_bone)
    return model


def build_fuser(args):
    extractor = build_extractor(args)
    return Fuser(args,extractor)


# if __name__ == "__main__":
#     model = build_fuser()
    
