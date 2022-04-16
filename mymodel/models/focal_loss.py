from typing import Optional, Sequence

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

def focal_loss(inputs: Tensor, targets: Tensor,alpha: float = 4.0,beta: float= .0,reduction: str='mean' ) -> Tensor:
    """
    inputs: A float tensor of arbitrary shape.
                The predictions for each example.
    targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
            (-1 for the negative class and 1 for the positive class).
    """
    loss = F.logsigmoid(inputs.mul(targets).mul(alpha).add(beta),dim = -1)
    loss = loss.sum(dim = -1).div(alpha) # b 

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss



    

