# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
import json

from bert import BertSC
from utils import BertTokenizer, VOCAB_NAME, MODEL_NAME, MODEL_NAME, BERT_CONFIG
from optimization import BertAdam
from preprocess.sequence_classification import load_dataset

from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
from fastNLP import AccuracyMetric

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='../bert_pytorch/tasks/SequenceClassification/glue_data/MRPC', # None,
                        type=str,
                        required=False, # True
                        help="The input squad_data dir. Should contain the .tsv files (or other squad_data files) for the task.")
    parser.add_argument("--bert_model", default='converted/base-uncased', type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='MRPC',
                        type=str,
                        required=False,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='tasks/SequenceClassification/mrpc_output',
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--vocab_size",
                        default=30522,
                        type=int,
                        help="The size of vocabulary.")
    parser.add_argument("--do_train", default=1, type=int,
                        # action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=1, type=int,
                        # action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=83,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    ###### config setting ######

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    ###### fastNLP.DataSet loading ######

    train_data, dev_data = load_dataset(args)

    ###### model initializing ######

    config = json.load(open(os.path.join(args.bert_model, BERT_CONFIG), "r"))
    model = BertSC(args.vocab_size, num_labels=args.num_labels, **config)
    model.load(os.path.join(args.bert_model, MODEL_NAME))
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    ###### ptimizer initializing ######

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.a_2', 'LayerNorm.b_2']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = args.num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        warmup=args.warmup_proportion,
        t_total=t_total
    )

    # optimizer = optim.Adam(params=optimizer_grouped_parameters,lr=args.learning_rate)

    ###### fastNLP.Trainer initializing ######
    trainer = Trainer(model=model,
                      train_data=train_data,
                      dev_data=dev_data,
                      loss=CrossEntropyLoss(pred="pred", target="target"),
                      metrics=AccuracyMetric(),
                      print_every=1,
                      optimizer=optimizer,
                      batch_size=args.train_batch_size,
                      n_epochs=args.num_train_epochs)

    # train our model
    trainer.train()


if __name__ == "__main__":
    main()
