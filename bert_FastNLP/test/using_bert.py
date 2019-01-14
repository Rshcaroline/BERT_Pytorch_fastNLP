# @Time    : 2019/1/11 2:01 PM
# @Author  : Shihan Ran
# @File    : using_bert.py
# @Software: PyCharm
# @license : Copyright(C), Fudan University
# @Contact : rshcaroline@gmail.com
# @Desc    :

import sys

sys.path.append("..")

import torch

from utils import BertTokenizer
from backbone import Bert
from bert import BertMLM

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('../converted/base-uncased')

# Tokenized input
text = "Who was Jim Aaron ? Jim Aaron was a puppeteer"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 6
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['who', 'was', 'jim', 'aaron', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# =================== Method 1: Load pre-trained BertModel (weights) ===================\
model = Bert(30522)
model.load('../converted/base-uncased/pytorch_model.bin')
model.eval()

# Predict hidden states features for each layer
encoded_layers, _ = model(tokens_tensor, segments_tensors)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
assert len(encoded_layers) == 12

# =================== Method 2: Load pre-trained BertForMaskedLM (weights) ===================
model = BertMLM(30522)
model.load('../converted/base-uncased/pytorch_model.bin')

model.eval()

# Predict all tokens
predictions = model(tokens_tensor, segments_tensors)

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print("predicted_token:", predicted_token)