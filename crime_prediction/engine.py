"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import torch
from utils import focal_loss,predict
from accelerate import Accelerator
from tqdm import tqdm
import logging
from sklearn.metrics import confusion_matrix,classification_report
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
        targets = batch.pop("labels").to(device)
        batch = batch.to(device)
        outputs = model(batch)
        loss = focal_loss(outputs, targets)
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

# eval only on one process
@torch.no_grad()
def evaluate(model: torch.nn.Module,data_loader: Iterable,
            accelerator:Accelerator, device, batch_size,metrics):
    model.eval()
    losses =[]

    for step, batch in enumerate(data_loader):
        targets = batch.pop("labels").to(device)
        batch = batch.to(device)
        outputs = model(batch)
        prediction = predict(outputs)
        loss = focal_loss(outputs, targets)
        # Batch

        loss_all,predictions, references = accelerator.gather((loss,prediction, targets))
        losses.append(loss_all)
        metrics.add_batch(predictions=predictions, references=references)
    eval_metric = metrics.compute()
    
    losses = torch.cat(losses)
    losses = losses[: len(data_loader)].tolist()
    eval_metric['loss'] = sum(losses)
    #ConfusionMatrixDisplay(cm, display_labels=pipeline.classes_)

    return eval_metric














