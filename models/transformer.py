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

        self.src_pad_index = src_pad_index
        self.trg_pad_index = trg_pad_index

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
        self.encoder_stack = nn.ModuleList(
            [EncoderLayer(d_model,
                          num_heads=num_heads,
                          dropout_rate=dropout_rate)
             for _ in range(num_layers)]
        )

        # Decoder
        self.decoder_stack = nn.ModuleList(
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
        """
        Use lower-triangular mask to prevent leftward information flow
        Fill upper triangle with negative infinity to zero out those values during softmax

        seq     weights      values          output
        0       [1 0 0]   [ --- a --- ]   [ a + 0 + 0 ]
        1       [1 1 0] * [ --- b --- ] = [ a + b + 0 ]
        2       [1 1 1]   [ --- c --- ]   [ a + b + c ]

        At seq=0, can only attend to seq=0
        At seq=1, can attend to both seq=0 and seq=1
        And so on...
        """

        # Generate masks
        batch = source.size(0)
        src_seq_len = source.size(-1)
        trg_seq_len = target.size(-1)

        flow_mask = torch.triu(     # Prevents leftward flow of information in target seq
            torch.ones(trg_seq_len, trg_seq_len, dtype=torch.bool, requires_grad=False),
            diagonal=1
        ).to(self.device)
        enc_mask = (source == self.src_pad_index)
        dec_mask = (target == self.trg_pad_index).unsqueeze(1) | flow_mask

        # Reshape to allow broadcasting to multi-headed tensors during attention
        enc_mask = enc_mask.reshape(batch, 1, 1, src_seq_len)
        dec_mask = dec_mask.reshape(batch, 1, trg_seq_len, trg_seq_len)

        # Encoder stack
        enc_out = self.src_embedding(source)
        for layer in self.encoder_stack:
            enc_out = layer(enc_out, enc_mask)

        # Decoder stack
        dec_out = self.trg_embedding(target)
        for layer in self.decoder_stack:
            dec_out = layer(dec_out, enc_out, enc_mask, dec_mask)

        # Final linear layer + softmax to get word probabilities
        return self.softmax(self.linear(dec_out))
