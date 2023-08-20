import config
import torch
from data import Dataset
from modules import Transformer
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm


WEIGHTS_PATH = 'experiments/en-de/08_19_2023/17_27_32/weights/609'

# Load data
dataset = Dataset(config.LANGUAGE_PAIR, batch_size=config.BATCH_SIZE)

# Load saved model
model = Transformer(
    config.D_MODEL,
    len(dataset.src_vocab),
    len(dataset.trg_vocab),
    dataset.src_vocab[dataset.pad_token],
    dataset.trg_vocab[dataset.pad_token]
)
model.load_state_dict(torch.load(WEIGHTS_PATH))

# Evaluate model on test set
with torch.no_grad():
    model.eval()
    valid_loss = 0
    num_batches = 0
    bleu_score = 0
    for data in tqdm(dataset.test_loader, desc='Test'):
        src = data['source'].to(model.device)
        trg = data['target'].to(model.device)

        predictions = model(src, trg[:, :-1])

        # Calculate BLEU score
        batch_size = predictions.size(0)
        batch_bleu = 0
        p_indices = torch.argmax(predictions, dim=-1)
        for i in range(batch_size):
            p_tokens = dataset.trg_vocab.lookup_tokens(p_indices[i].tolist())
            t_tokens = dataset.trg_vocab.lookup_tokens(trg[i, 1:].tolist())

            # Filter out special tokens
            p_tokens = list(filter(lambda x: '<' not in x, p_tokens))
            t_tokens = list(filter(lambda x: '<' not in x, t_tokens))

            if len(p_tokens) > 0 and len(t_tokens) > 0:
                batch_bleu += sentence_bleu([t_tokens], p_tokens)

        bleu_score += batch_bleu / batch_size
        num_batches += 1
        del src, trg

    print('\nBLEU score:', bleu_score / num_batches * 100)
