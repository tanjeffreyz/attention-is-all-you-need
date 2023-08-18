import torch
from torch import nn
from .interfaces import Module
from .embedding import Embedding
from .encoder import EncoderLayer
from .decoder import DecoderLayer


class Transformer(Module):
    def __init__(self,
                 d_model,
                 src_vocab_len,
                 trg_vocab_len,
                 src_pad_index,
                 trg_pad_index,
                 num_heads=8,
                 num_layers=6,
                 dropout_rate=0.1,
                 seed=20230815):
        super().__init__()

        # Manually seed to keep embeddings consistent across loads
        torch.manual_seed(seed)

        # Embeddings, pass in pad indices to prevent <pad> from contributing to gradient
        self.src_embedding = Embedding(d_model,
                                       src_vocab_len,
                                       src_pad_index,
                                       dropout_rate=dropout_rate)
        self.trg_embedding = Embedding(d_model,
                                       trg_vocab_len,
                                       trg_pad_index,
                                       dropout_rate=dropout_rate)

        # Encoder
        self.encoder = nn.Sequential(
            *[EncoderLayer(d_model,
                           num_heads=num_heads,
                           dropout_rate=dropout_rate)
              for _ in range(num_layers)]
        )

        # Decoder
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model,
                          num_heads=num_heads,
                          dropout_rate=dropout_rate)
             for _ in range(num_layers)]
        )

        # Final layer to project embedding to target vocab word probability distribution
        self.linear = nn.Linear(d_model, trg_vocab_len)
        self.softmax = nn.Softmax(dim=-1)

        # Move to GPU if possible
        self.to(self.device)

        # Re-seed afterward to allow shuffled data
        torch.seed()

    def forward(self, source, target):
        # Encoder stack
        src_embedding = self.src_embedding(source)
        enc_out = self.encoder(src_embedding)

        # Decoder stack
        dec_out = self.trg_embedding(target)
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out)

        # Final linear layer + softmax to get word probabilities
        return self.softmax(self.linear(dec_out))
