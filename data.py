import config
import spacy
import torch
from torch import nn
from torchtext.datasets import multi30k, Multi30k
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader


# Patch Multi30K dataset, PyTorch's Google Drive link is no longer active
multi30k.URL['train'] = 'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz'
multi30k.URL['valid'] = 'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz'
multi30k.URL['test'] = 'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz'

# Data preprocessing helpers
pipelines = {
    'en': spacy.load('en_core_web_sm'),
    'de': spacy.load('de_core_news_sm')
}

src_lang, trg_lang = config.LANGUAGE_PAIR
assert src_lang in pipelines, 'Unrecognized source language'
assert trg_lang in pipelines, 'Unrecognized target language'


def tokenize(pair):
    """Splits source and target sentences into tokens based on the configured language pair"""

    src, trg = pair
    src_tokens = [config.SOS] + [x.text.lower() for x in pipelines[src_lang].tokenizer(src)] + [config.EOS]
    trg_tokens = [config.SOS] + [y.text.lower() for y in pipelines[trg_lang].tokenizer(trg)] + [config.EOS]
    return src_tokens, trg_tokens


def encode(pair):
    """Replaces tokens with their corresponding vocabulary indices"""

    src, trg = pair
    return (
        torch.tensor(src_vocab.lookup_indices(src)),
        torch.tensor(trg_vocab.lookup_indices(trg))
    )


# Load datasets
train_data, _, test_data = Multi30k(root='data', language_pair=config.LANGUAGE_PAIR)

# Build vocabs from dataset
src_vocab = build_vocab_from_iterator(
    map(lambda x: x[0], train_data.map(tokenize)),
    specials=[config.SOS, config.EOS]
)
src_vocab.set_default_index(-1)

trg_vocab = build_vocab_from_iterator(
    map(lambda x: x[1], train_data.map(tokenize)),
    specials=[config.SOS, config.EOS]
)
trg_vocab.set_default_index(-1)

# Initialize embeddings with same seed every time
torch.manual_seed(20230815)
src_embedding = nn.Embedding(len(src_vocab), config.D_MODEL)
trg_embedding = nn.Embedding(len(trg_vocab), config.D_MODEL)
torch.seed()        # Reseed afterward b/c want shuffled data

# Tokenize, encode, and batch the data
train_processed = (
    train_data
    .map(tokenize)
    .map(encode)
    .batch(config.BATCH_SIZE)
    .rows2columnar(config.LANGUAGE_PAIR)
)
test_processed = (
    test_data
    .map(tokenize)
    .map(encode)
    .batch(config.BATCH_SIZE)
    .rows2columnar(config.LANGUAGE_PAIR)
)
train_loader = DataLoader(train_processed, batch_size=None, shuffle=True)
test_loader = DataLoader(test_processed, batch_size=None, shuffle=False)
