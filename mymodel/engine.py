"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch
from utils import focal_loss
from accelerate import Accelerator
from tqdm import tqdm
import logging
logger = logging.getLogger("train model")


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, accelerator:Accelerator,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    logger.info('Epoch: [{}]'.format(epoch))
    tr_loss = 0
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    steps = tqdm(data_loader,ncols = 110)
    for batch in steps:
        targets = batch.pop("classification").to(device)
        batch = batch.to(device)
        _,outputs = model(batch)
        loss,_ = focal_loss(outputs, targets,device)
        loss_value = loss.item()
        tr_loss += loss_value

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        accelerator.backward(loss)
        if max_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        steps.set_description("Epoch {}, Loss {:.7f}".format(epoch + 1, loss_value))

    # gather the stats from all processes
    return tr_loss


@torch.no_grad()
def evaluate(model: torch.nn.Module,data_loader: Iterable,
            accelerator:Accelerator, device, batch_size):
    model.eval()
    losses = []
    predictions = []
    labels = []
    for step, batch in enumerate(data_loader):
        targets = batch.pop("classification").to(device)
        batch = batch.to(device)
        _,outputs = model(batch)
        loss,prediction = focal_loss(outputs, targets,device,eval = True)
        # Batch
        losses.append(accelerator.gather(loss.repeat(batch_size)))
        predictions.append(accelerator.gather(prediction))
        labels.append(accelerator.gather(targets))
    
    losses = torch.cat(losses)
    losses = losses[: len(data_loader)]
    predictions = torch.cat(predictions)
    predictions = predictions[: len(data_loader)]
    labels = torch.cat(labels)
    labels = labels[: len(data_loader)]
    TP = ((predictions.detach() == 1) & (labels.detach() == 1)).cpu().sum().item()
    # TN    predict 和 label 同时为0
    TN = ((predictions.detach() == 0) & (labels.detach() == -1)).cpu().sum().item()
    # FN    predict 0 label 1
    FN = ((predictions.detach() == 0) & (labels.detach() == 1)).cpu().sum().item()
    # FP    predict 1 label 0
    FP = ((predictions.detach() == 1) & (labels.detach() == -1)).cpu().sum().item()


    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return {"precision": p,"recall": r ,"F1": F1,"accuarcy":acc}













