# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader
from models import build_model
from dataset import build_dataset
from engine import evaluate, train_one_epoch
from transformers import default_data_collator
from accelerate import Accelerator


def get_args_parser():
    parser = argparse.ArgumentParser('set similar case matching', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=3e-6, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr_drop', default=6, type=int)
    parser.add_argument('--clip_max_norm', default=1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--stage', default=2, type=int)
    parser.add_argument('--num_labels', default=60, type=int)
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--back_bone', default='bert-base-multilingual-uncased', type=str,
                        help="Name of the bert backbone to use")


    # Loss
    # * Loss coefficients
    parser.add_argument('--focal_alpha', default=4, type=float)
    parser.add_argument('--focal_beta', default=0, type=float)


    # dataset parameters

    parser.add_argument('--dataset', default='scm')
    parser.add_argument('--scm_path', type=str)
    parser.add_argument('--legal_element_path', type=str)

    parser.add_argument('--output_dir', default='/home/huijie/legal/mymodel/saved_model/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int,
                        help='local rank')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    #logging
    logger = logging.getLogger("train model")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(os.path.join(args.output_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)


    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(args)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ] #TODO check the backbone name
    logger.info(
            "named_parameters: {}".format(
                json.dumps(model.named_parameters(), indent=4, sort_keys=True)
            )
        )

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    accelerator = Accelerator()
    with accelerator.main_process_first():
        dataset_train,dataset_val = build_dataset(args)
    train_dataloader = DataLoader(dataset_train,args.batch_size,collate_fn=default_data_collator)
    eval_dataloader = DataLoader(dataset_val, args.batch_size,collate_fn=default_data_collator) 
    # TODO check defult dataloader

    
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )


    output_dir = Path(args.output_dir)
    # if args.resume:
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     model.load_state_dict(checkpoint['model'])
    #     if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #         args.start_epoch = checkpoint['epoch'] + 1

    # if args.eval:
    #     test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                           data_loader_val, base_ds, device, args.output_dir)
    #     if args.output_dir:
    #         utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
    #     return
    logger.info("***** Running training *****")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, train_dataloader, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        # if args.output_dir and (epoch + 1) % 3 == 0:
        #     checkpoint_path= (output_dir / f'checkpoint{epoch:04}.pth')
        #     utils.save_on_master({
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch,
        #         'args': args,
        #     }, checkpoint_path)

        test_stats = evaluate(
            model, eval_dataloader, accelerator,device, args.batch_size
        )
        logger.info(test_stats)

        # if args.output_dir and utils.is_main_process():
        #     # for evaluation logs
        #     if coco_evaluator is not None:
        #         (output_dir / 'eval').mkdir(exist_ok=True)
        #         if "bbox" in coco_evaluator.coco_eval:
        #             filenames = ['latest.pth']
        #             if epoch % 50 == 0:
        #                 filenames.append(f'{epoch:03}.pth')
        #             for name in filenames:
        #                 torch.save(coco_evaluator.coco_eval["bbox"].eval,
        #                            output_dir / "eval" / name)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SCM training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
