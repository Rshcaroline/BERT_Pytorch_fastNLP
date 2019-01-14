import torch
import torch.nn as nn
from fastNLP.modules.encoder import Embedding
from fastNLP.modules.other_modules import LayerNormalization


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_position=512, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param max_position: max position value
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = Embedding(nums=vocab_size, dims=embed_size, dropout=0)
        self.position = Embedding(nums=max_position, dims=embed_size, dropout=0)
        self.segment = Embedding(nums=2, dims=embed_size, dropout=0)
        self.dropout = nn.Dropout(p=dropout)
        self.LayerNorm = LayerNormalization(embed_size)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label=None):
        seq_length = sequence.size(1)
        positions = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        positions = positions.unsqueeze(0).expand_as(sequence)

        if segment_label is None:
            segment_label = torch.zeros_like(positions)

        embeddings = self.LayerNorm(self.token(sequence) + self.position(positions) + self.segment(segment_label))
        return self.dropout(embeddings)