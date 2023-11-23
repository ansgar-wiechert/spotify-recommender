from torch.nn import Module

class Transformer(Module):
    """Combines encoder-decoder architecture into single model.

    Args:
        Module: _description_
    """
    def __init__(self, encoder, decoder, src_embed, target_embed, output_layer):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.output_layer = output_layer

    def forward(self):
        """Apply encoding-decoding step.
        """
        pass

    def encode(self):
        """Compute encoding representation of source embeding.
        """
        pass

    def decode(self):
        """Compute decoding representation of target embeding and source embeding.
        """
        pass

