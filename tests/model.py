import unittest
import torch
from models.embedding import Embedding
from models.attention import MultiHeadAttention
from models.encoder import EncoderLayer
from models.decoder import DecoderLayer
from models.feed_forward import FeedForwardNetwork


class TestEmbedding(unittest.TestCase):
    def test_embedding(self):
        tokens = torch.ones((64, 30), dtype=torch.int).to(Embedding.device)
        embedding = Embedding(3000, 512).to(Embedding.device)
        self.assertEqual(
            embedding(tokens).size(),
            torch.Size([64, 30, 512])
        )

    def test_positional_encoding(self):
        tokens = torch.ones((64, 30), dtype=torch.int).to(Embedding.device)
        embedding = Embedding(3000, 512).to(Embedding.device)
        self.assertEqual(
            embedding.positional_encoding(tokens).size(),
            torch.Size([30, 512])
        )


class TestAttention(unittest.TestCase):
    def test_scaled_dot_product_attention(self):
        x = torch.rand((64, 8, 30, 64))
        attention = MultiHeadAttention(512)
        self.assertEqual(
            attention.scaled_dot_product_attention(x, x, x).size(),
            torch.Size([64, 8, 30, 64])
        )

    def test_multi_head_attention(self):
        x = torch.rand((64, 30, 512))
        attention = MultiHeadAttention(512)
        self.assertEqual(
            attention(x, x, x).size(),
            torch.Size([64, 30, 512])
        )

    def test_split_heads(self):
        x = torch.rand((64, 30, 512))
        attention = MultiHeadAttention(512)
        self.assertEqual(
            attention.split_heads(x).size(),
            torch.Size([64, 8, 30, 64])
        )

    def test_merge_heads(self):
        x = torch.rand((64, 8, 30, 64))
        attention = MultiHeadAttention(512)
        self.assertEqual(
            attention.merge_heads(x).size(),
            torch.Size([64, 30, 512])
        )


class TestTransformer(unittest.TestCase):
    def test_encoder_layer(self):
        x = torch.rand((64, 30, 512))
        encoder = EncoderLayer(512)
        self.assertEqual(
            encoder(x).size(),
            torch.Size([64, 30, 512])
        )

    def test_decoder_layer(self):
        x = torch.rand((64, 30, 512)).to(DecoderLayer.device)
        decoder = DecoderLayer(512).to(DecoderLayer.device)
        self.assertEqual(
            decoder(x, x).size(),
            torch.Size([64, 30, 512])
        )

    def test_feed_forward_network(self):
        x = torch.rand((64, 30, 512))
        ffn = FeedForwardNetwork(512)
        self.assertEqual(
            ffn(x).size(),
            torch.Size([64, 30, 512])
        )


if __name__ == '__main__':
    unittest.main()
