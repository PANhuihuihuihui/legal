import torch
from torch import FUSE_ADD_RELU, nn
import torch.nn.functional as F

from extractor import build_fuser,build_extractor

class Matcher(nn.Module):
    def __init__(self,args,fuser) -> None:
        super().__init__()
        self.fuser = fuser
        self.fc = nn.Linear(args.hidden_state,args.candidate_len,bias=False)
        # self.init_weights()
    def forward(self,archer,candidates):
        archer = self.fuser(archer) # b E
        Vqi = torch.zeros(archer.shape)  # b E
        for candidate in candidates:
            di = self.fuser(candidate)
            Vqi = Vqi.add(archer.sub(di).div(len(candidates)))
        archer = F.softmax(self.fc(Vqi),dim= 1)
        return archer
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)


def build_model(args):
    if args.stage ==1:
        model = build_extractor(args)
        return model
    else:
        model = build_fuser(args)
        model = Matcher(args,model)
    

    