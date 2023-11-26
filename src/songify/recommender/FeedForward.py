import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.nn import Linear
import copy
import torch

class OutputLayer(nn.Module):
    """Final output layer generating output sequence.

    Args:
        nn: _description_
    """
    def __init__(self, d_model, max_sequence):
        super().__init__()
        self.d_model = d_model # not needed
        self.max_sequence = max_sequence # not needed
        self.linear = Linear(d_model, max_sequence)

    def forward(self, x):
        """Apply linear layer and softmax.

        Args:
            x: input token.

        Returns:
            x: output token.
        """
        x = log_softmax(self.linear(x), dim=-1)
        return x

class LayerNorm(nn.Module):
    """_summary_

    Args:
        nn: _description_
    """
    def __init__(self, n_features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_features))
        self.beta = nn.Parameter(torch.ones(n_features))
        self.eps = eps

    def forward(self, x):
        """Apply forward pass of layer normalization.

        Args:
            x: _description_

        Returns:
            _description_
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.gamma * (x - mean) / (std + self.eps) + self.beta

        return x

class SublayerConnection(nn.Module):
    """_summary_

    Args:
        nn: _description_
    """
    def __init__(self, input_dim, p_dropout):
        super().__init__()
        self.norm = LayerNorm(input_dim)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, sublayer):
        """_summary_

        Args:
            x: _description_
            sublayer: _description_

        Returns:
            _description_
        """
        x = x + self.dropout(sublayer(self.norm(x)))
        return x

def clone_layers(module, n):
    """_summary_

    Args:
        module: _description_
        n: _description_
    """
    copies = nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
    return copies
