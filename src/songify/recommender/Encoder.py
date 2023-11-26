import torch.nn as nn
from songify.Recommender.FeedForward import SublayerConnection, clone_layers

class Encoder(nn.Module):
    """_summary_

    Args:
        nn: _description_
    """
    def __init__(self, layer, N):
        super().__init__()
        self.layer = clone_layers(Encoder, N)
        

class EncoderLayer(nn.Module):
    """_summary_

    Args:
        nn: _description_
    """
    def __init__(self, size, attention, linear_layer, dropout):
        super().__init__()
        self.size = size
        self.attention = attention
        self.linear_layer = linear_layer
        self.sublayer = clone_layers(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        """_summary_

        Args:
            x: _description_
            mask: _description_
        """
        pass