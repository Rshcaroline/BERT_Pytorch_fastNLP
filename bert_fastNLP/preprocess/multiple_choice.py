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

import os
import argparse
import random
from tqdm import tqdm, trange
import csv

from utils import BertTokenizer, VOCAB_NAME, MODEL_NAME

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class SwagExample(object):
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 swag_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label = None):
        self.swag_id = swag_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"swag_id: {self.swag_id}",
            f"context_sentence: {self.context_sentence}",
            f"start_ending: {self.start_ending}",
            f"ending_0: {self.endings[0]}",
            f"ending_1: {self.endings[1]}",
            f"ending_2: {self.endings[2]}",
            f"ending_3: {self.endings[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def read_swag_examples(input_file, is_training):
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = list(reader)

    if is_training and lines[0][-1] != 'label':
        raise ValueError(
            "For training, the input file must contain a label column."
        )

    examples = [
        SwagExample(
            swag_id = line[2],
            context_sentence = line[4],
            start_ending = line[5], # in the swag dataset, the
            # common beginning of each
            # choice is stored in "sent2".
            ending_0 = line[7],
            ending_1 = line[8],
            ending_2 = line[9],
            ending_3 = line[10],
            label = int(line[11]) if is_training else None
        ) for line in lines[1:] # we skip the line with the column names
    ]

    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a squad_data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for example_index, example in enumerate(examples):
        context_tokens = tokenizer.tokenize(example.context_sentence)
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]
            ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label

        features.append(
            InputFeatures(
                example_id = example.swag_id,
                choices_features = choices_features,
                label = label
            )
        )

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def load_dataset(args):

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    args.num_train_steps = None

    train_examples = read_swag_examples(os.path.join(args.data_dir, 'train.csv'), is_training = True)
    print(len(train_examples), args.train_batch_size, args.gradient_accumulation_steps, args.num_train_epochs)
    args.num_train_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # training dataset
    train_features = convert_examples_to_features(
        train_examples, tokenizer, args.max_seq_length, True)

    all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)

    # dev dataset
    eval_examples = read_swag_examples(os.path.join(args.data_dir, 'val.csv'), is_training=True)
    eval_features = convert_examples_to_features(
        eval_examples, tokenizer, args.max_seq_length, True)

    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
    dev_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)

    return train_data, dev_data