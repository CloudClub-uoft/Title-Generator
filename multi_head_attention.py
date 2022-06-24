from torch import nn
import torch

from scaled_dot_attention import attention

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, input_dim, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dim_heads = embed_dim // num_heads     # dim_heads aka d_k

        self.q_lin = nn.Linear(input_dim, embed_dim)
        self.k_lin = nn.Linear(input_dim, embed_dim)
        self.v_lin = nn.Linear(input_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        num_heads, dim_heads = self.num_heads, self.dim_heads

        q = self.q_lin(q).reshape(batch_size, -1, num_heads, dim_heads).transpose(1, 2)
        k = self.k_lin(k).reshape(batch_size, -1, num_heads, dim_heads).transpose(1, 2)
        v = self.v_lin(v).reshape(batch_size, -1, num_heads, dim_heads).transpose(1, 2)

        scores = attention(q, k, v, dim_heads, mask=mask, dropout=self.dropout)
        
        scores = scores.transpose(1, 2).contiguous().reshape(batch_size, -1, self.embed_dim)

        output = self.out_proj(scores)

        return output

if __name__ == "__main__":
    embed_dim = 512
    num_heads = 8

    q = torch.tensor([[0, 10, 0]], dtype=torch.float32)
    k = torch.tensor([[0, 10, 0]], dtype=torch.float32)
    v = torch.tensor([[0, 10, 0]], dtype=torch.float32)
    input_dim = 3

    mh = MultiHeadAttention(num_heads, embed_dim, input_dim)
    scores = mh.forward(q, k, v)
    print(scores)