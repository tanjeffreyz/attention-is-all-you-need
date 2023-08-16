import torch
from torch import nn


class Module(nn.Module):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
