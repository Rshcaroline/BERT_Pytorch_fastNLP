import torch
from torch import nn

from embedding import BERTEmbedding
from utils import GeLU
from fastNLP.modules.encoder import TransformerEncoder as Transformer


class Bert(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, dropout=dropout)

        # multi-layers transformer blocks, deep network
        self.transformer = Transformer(
            num_layers=n_layers, 
            num_atte=attn_heads,
            input_size=hidden,
            intermediate_size=self.feed_forward_hidden,
            key_size=hidden,
            output_size=hidden,
            activate=GeLU,
            dropout=dropout,
        )

        # Pooling layer
        self.pooler = nn.Linear(hidden, hidden)
        self.activation = nn.Tanh()

    def bert_forward(self, x, segment_info, mask=None, all_output=True):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        if mask is None:
            mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1).float()
        else:
            mask = (mask > 0).unsqueeze(1).repeat(1, mask.size(1), 1).unsqueeze(1).float()

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        encoder_layers = self.transformer.forward(x, mask, all_output=all_output)

        last_layer = encoder_layers[-1]

        cls_token = last_layer[:, 0]
        pool_output = self.pooler(cls_token)
        pool_output = self.activation(pool_output)

        if all_output:
            return encoder_layers, pool_output

        else:
            return last_layer, pool_output

    def forward(self, x, segment_info, mask=None):
        return self.bert_forward(x, segment_info, mask)

    def load(self, load_path):
        # load pretrained
        pretrained_dict = torch.load(load_path)
        model_dict = self.state_dict()
        # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        print("loading pre-trained weights from %s" % load_path)
