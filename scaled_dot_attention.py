import torch
from math import sqrt
import torch.nn.functional as F

def attention(q, k, v, d_k, mask=None, dropout=None):
    scaled_dot = torch.matmul(q, k.transpose(-2, -1)) / sqrt(d_k)
    if mask is not None:
        scaled_dot = scaled_dot.masked_fill(mask == 0, -1e9)
    scaled_dot = F.softmax(scaled_dot, dim=-1)
    if dropout is not None:
        scaled_dot = dropout(scaled_dot)
    output =  torch.matmul(scaled_dot, v)
    return output