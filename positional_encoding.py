import torch
from torch import nn
from math import log

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, input_dim):
        """
        embed_dim: num of expected features in input (same as d_model)
        input_dim: length of sequence
        """
        super().__init__()

        encod = torch.zeros(input_dim, embed_dim)

        position = torch.arange(0, input_dim, dtype=torch.float).unsqueeze(1)   # numerator

        i = torch.arange(0, embed_dim, 2, dtype=torch.float)

        denom = torch.exp(log(10000.0) * i / embed_dim)

        encod[ : , 0::2] = torch.sin(position / denom)
        encod[ : , 1::2] = torch.cos(position / denom)
        encod.unsqueeze(0)

        self.pe = encod


    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x