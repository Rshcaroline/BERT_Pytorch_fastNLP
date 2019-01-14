import torch

from torch import nn

from ..aggregator.attention import MultiHeadAtte
from ..other_modules import LayerNormalization


class TransformerEncoder(nn.Module):
    class SubLayer(nn.Module):
        def __init__(self, input_size, output_size, intermediate_size, key_size, value_size, num_atte, activate=torch.nn.ReLU, dropout=0.0):
            super(TransformerEncoder.SubLayer, self).__init__()
            # TODO: change MultiHeadAtte
            
            self.atte = MultiHeadAtte(input_size, output_size, key_size, num_atte, dropout)
            self.intermediate = nn.Sequential(
                nn.Linear(output_size, intermediate_size),
                activate(),
            )
            self.output = nn.Sequential(
                nn.Linear(intermediate_size, output_size),
                nn.Dropout(dropout)
            )
            self.LayerNorm = LayerNormalization(output_size)

        def forward(self, input, seq_mask):
            attention = self.atte(input, seq_mask)
            intermediate = self.intermediate(attention)
            output = self.output(intermediate)
            return self.LayerNorm(output + attention)

    def __init__(self, num_layers, **kargs):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([self.SubLayer(**kargs) for _ in range(num_layers)])

    def forward(self, x, seq_mask=None, all_output=True):
        all_encoder_layers = []
        for layer in self.layers:
            x = layer(x, seq_mask)
            if all_output:
                all_encoder_layers.append(x)
        if not all_output:
            all_encoder_layers.append(x)
        return all_encoder_layers
