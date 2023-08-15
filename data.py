from torchtext.datasets import multi30k, Multi30k
from torch.utils.data import DataLoader

multi30k.URL['train'] = 'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz'
multi30k.URL['valid'] = 'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz'
multi30k.URL['test'] = 'https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt16_task1_test.tar.gz'

train_data, _, test_data = Multi30k(root='data', language_pair=('de', 'en'))

train_loader = DataLoader(train_data, )
