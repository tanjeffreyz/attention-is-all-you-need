import config
import spacy
from torchtext.functional import to_tensor
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
    return src_vocab.lookup_indices(src), trg_vocab.lookup_indices(trg)


def pad(batch):
    """
    Pads all sequences in the batch to have same length. Pads using the EOS token because if <eos> is reached, model
    should insist sequence has finished and output <eos> even if prompted further.
    """

    batch['source'] = to_tensor(batch['source'], padding_value=src_vocab[config.EOS])
    batch['target'] = to_tensor(batch['target'], padding_value=trg_vocab[config.EOS])
    return batch


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

# Tokenize, encode, and batch the data
columns = ['source', 'target']
train_processed = (
    train_data
    .map(tokenize)
    .map(encode)
    .batch(config.BATCH_SIZE)
    .rows2columnar(columns)
    .map(pad)
)
test_processed = (
    test_data
    .map(tokenize)
    .map(encode)
    .batch(config.BATCH_SIZE)
    .rows2columnar(columns)
    .map(pad)
)
train_loader = DataLoader(train_processed, batch_size=None, shuffle=True)
test_loader = DataLoader(test_processed, batch_size=None, shuffle=False)
