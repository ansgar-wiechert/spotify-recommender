import torch.nn as nn
import torch

from torch.nn.functional import log_softmax, pairwise_distance
from torch.nn import Linear

class OutputLayer(nn.Module):
    """Final output layer generating output sequence. Applies linear layer and softmax.

    Args:
        nn: _description_
    """
    def __init__(self, d_model, max_sequence):
        super(OutputLayer, self).__init__()
        self.d_model = d_model # not needed
        self.max_sequence = max_sequence # not needed
        self.linear = Linear(d_model, max_sequence)

    def forward(self, x):
        return 