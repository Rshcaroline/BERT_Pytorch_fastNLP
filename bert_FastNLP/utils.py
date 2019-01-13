import torch

from torch import nn

class GeLU(nn.Module):
    """ 
    Implementation of the gelu activation function.
    """
    
    def forward(self, x):
        """
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))