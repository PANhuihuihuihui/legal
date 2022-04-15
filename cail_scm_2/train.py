import logging
import os

import argparse
from xml.etree.ElementTree import TreeBuilder

from model import HyperParameters, BertModelTrainer


def get_args_parser():
    # Training
    parser = argparse.ArgumentParser('SCM experiment', add_help=False)
    parser.add_argument('--batch-size', default=12, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--k-fold', default=1, type=int)
    parser.add_argument('--model-name', default='scm', type=str, metavar='PATH',
                        help='Path of model to train')
    parser.add_argument('--bert-model', default='bert-base-multilingual-uncased', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--bert-path', default='/home/huijie/legal/huggface/multilingual_cail', type=str, metavar='PATH',
                        help='Path of model to train')


    return parser

parser = argparse.ArgumentParser('SCM experiment', parents=[get_args_parser()])
args = parser.parse_args()

logger = logging.getLogger("train model")
logger.setLevel(logging.INFO)
logger.propagate = False
logging.getLogger("transformers").setLevel(logging.ERROR)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

MODEL_DIR = "/home/huijie/legal/cail_scm_2/saved_model/"+args.model_name
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
fh = logging.FileHandler(os.path.join(MODEL_DIR, "train.log"), encoding="utf-8")
fh.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

if __name__ == "__main__":
    # TRAINING_DATASET = "data/train/input.txt"  # for quick dev
    TRAINING_DATASET = "data/raw/CAIL2019-SCM-big/SCM_5k.json"

    test_input_path = "data/test/input.txt"
    test_ground_truth_path = "data/test/ground_truth.txt"

    config = {
        "max_length": 512,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": 2e-5,
        "fp16": False,
        "fp16_opt_level": "O1",
        "max_grad_norm": 1.0,
        "warmup_steps": 0.05,
    }
    hyper_parameter = HyperParameters()
    hyper_parameter.__dict__ = config
    algorithm = "BertForSimMatchModel"

    trainer = BertModelTrainer(
        TRAINING_DATASET,
        args.bert_model,
        args.bert_path,
        hyper_parameter,
        algorithm,
        test_input_path,
        test_ground_truth_path,
    )
    trainer.train(MODEL_DIR, args.k_fold)