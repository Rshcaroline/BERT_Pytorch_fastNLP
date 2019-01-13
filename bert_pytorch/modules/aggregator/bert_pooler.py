# @Time    : 2019/1/10 10:57 PM
# @Author  : Shihan Ran
# @File    : bert_pooler.py
# @Software: PyCharm
# @license : Copyright(C), Fudan University
# @Contact : rshcaroline@gmail.com
# @Desc    :

import torch.nn as nn


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
