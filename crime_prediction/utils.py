

import torch
from torch import Tensor
from torch.nn import functional as F

focal_loss = torch.hub.load(
	'adeelh/pytorch-multi-class-focal-loss',
	model='FocalLoss',
	# alpha=torch.tensor([.75, .25]),
	gamma=2,
	reduction='mean',
	force_reload=False
)
def predict(outputs):
    return torch.argmax(F.softmax(outputs,dim =-1),dim =-1 )

# def focal_loss(inputs: Tensor, targets: Tensor,device,
#                 alpha: float = 4.0,beta: float= .0,reduction: str='mean',
#                 eval: bool = False) -> Tensor:
#     """
#     inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#     targets: A float tensor with the same shape as inputs. Stores the binary
#                 classification label for each element in inputs
#             (-1 for the negative class and 1 for the positive class).
#     """

#     prediction = None
#     if eval:
#         prediction = (torch.sigmoid(inputs.mul(alpha).add(beta)) > 0.5).float()
#         # TP += ((inputs == 1) & (targets.detach() == 1)).cpu().sum()
#         # # TN    predict 和 label 同时为0
#         # TN += ((inputs == 0) & (targets.detach() == -1)).cpu().sum()
#         # # FN    predict 0 label 1
#         # FN += ((inputs == 0) & (targets.detach() == 1)).cpu().sum()
#         # # FP    predict 1 label 0
#         # FP += ((inputs == 1) & (targets.detach() == -1)).cpu().sum()
#         # loss_matrix = torch.tensor([TP,TN,FN,FP]).to(device)
#     # print(inputs.shape,targets.shape)
#     loss = F.logsigmoid(inputs.mul(targets).mul(alpha).add(beta))
#     loss = loss.sum(dim = -1).div(-alpha) # b 
#     if reduction == 'mean':
#         loss = loss.mean()
#     elif reduction == 'sum':
#         loss = loss.sum()
#     return loss, prediction
 
def eval_matrix(inputs: Tensor, targets: Tensor,alpha: float = 4.0,beta: float= .0):
    inputs = (F.sigmoid(inputs.mul(alpha).add(beta)) > 0.5).float()
    TP,TN,FN,FP = 0,0,0,0
    # TP    predict 和 label 同时为1
    TP += ((inputs == 1) & (targets.detach() == 1)).cpu().sum()
    # TN    predict 和 label 同时为0
    TN += ((inputs == 0) & (targets.detach() == -1)).cpu().sum()
    # FN    predict 0 label 1
    FN += ((inputs == 0) & (targets.detach() == 1)).cpu().sum()
    # FP    predict 1 label 0
    FP += ((inputs == 1) & (targets.detach() == -1)).cpu().sum()


# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#     return 



