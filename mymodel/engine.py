"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
from numpy import True_

import torch
from utils import focal_loss,eval_matrix
from accelerate import Accelerator
from tqdm import tqdm
logger = logging.getLogger("train model")


def train_one_epoch(model: torch.nn.Module, accelerator:Accelerator,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    logger.info('Epoch: [{}]'.format(epoch))
    tr_loss = 0
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    steps = tqdm(data_loader,ncols = 110)
    for step,batch in enumerate(steps):
        targets = batch.pop("classification").to(device)
        batch = batch.to(device)
        outputs = model(**batch)
        loss = focal_loss(outputs, targets)
        loss_value = loss.item()
        tr_loss += loss_value

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        accelerator.backward(loss)
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        steps.set_description("Epoch {}, Loss {:.7f}".format(epoch + 1, loss.item()))

    # gather the stats from all processes
    return tr_loss


@torch.no_grad()
def evaluate(model: torch.nn.Module,data_loader: Iterable,
            accelerator:Accelerator, device, batch_size):
    model.eval()
    losses = []
    loss_matrixs = []
    for step, batch in enumerate(data_loader):
        targets = batch.pop("classification").to(device)
        batch = batch.to(device)
        outputs = model(**batch)
        loss,loss_matrix = focal_loss(outputs, targets,eval = True)
        # Batch 
        losses.append(accelerator.gather(loss.repeat(batch_size)))
        loss_matrixs.append(accelerator.gather(loss_matrix.repeat(batch_size)))
    
    losses = torch.cat(losses)
    losses = losses[: len(data_loader)]
    loss_matrixs = torch.cat(loss_matrixs)
    loss_matrixs = loss_matrixs[: len(data_loader)]
    TP,TN,FN,FP = loss_matrixs.sum(dim =0).item()
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return {"precision": p,"recall": r ,"F1": F1,"accuarcy":acc}













