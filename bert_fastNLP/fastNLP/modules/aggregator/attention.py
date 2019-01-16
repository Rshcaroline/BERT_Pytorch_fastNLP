import math

import torch
from torch import nn

from fastNLP.modules.utils import mask_softmax
from fastNLP.modules.other_modules import LayerNormalization


class Attention(torch.nn.Module):
    def __init__(self, normalize=False):
        super(Attention, self).__init__()
        self.normalize = normalize

    def forward(self, query, memory, mask):
        similarities = self._atten_forward(query, memory)
        if self.normalize:
            return mask_softmax(similarities, mask)
        return similarities

    def _atten_forward(self, query, memory):
        raise NotImplementedError


class DotAtte(nn.Module):
    def __init__(self, key_size, value_size):
        super(DotAtte, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.scale = math.sqrt(key_size)

    def forward(self, Q, K, V, seq_mask=None):
        """

        :param Q: [batch, seq_len, key_size]
        :param K: [batch, seq_len, key_size]
        :param V: [batch, seq_len, value_size]
        :param seq_mask: [batch, seq_len]
        """
        output = torch.matmul(Q, K.transpose(1, 2)) / self.scale
        if seq_mask is not None:
            output.masked_fill_(seq_mask.lt(1), -float('inf'))
        output = nn.functional.softmax(output, dim=2)
        return torch.matmul(output, V)

# input_size, output_size, key_size, value_size, num_atte

class MultiHeadAtte(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_atte, dropout):
        super(MultiHeadAtte, self).__init__()
        if hidden_size % num_atte != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_atte))
        self.num_attention_heads = num_atte
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNormalization(hidden_size, eps=1e-12)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.dense(context_layer)
        output = self.dropout(output)
        return self.LayerNorm(hidden_states + output)

