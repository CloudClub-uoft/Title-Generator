import torch
import torch.nn as nn

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=512):
        super().__init__()
        # embed_dim: embedding dimension (usually 1024 or 512)
        self.embed_dim = embed_dim
        self.embed_matrix = torch.empty([vocab_size, embed_dim])

        nn.init.xavier_normal_(self.embed_matrix)
        self.embed_matrix = nn.Parameter(self.embed_matrix)
        self.embed_matrix = self.embed_matrix.to(torch.float)
        # seq len x vocab_size, vocab_size x embed_dim
        # embedding matrix dimensions: number of words in vocab x embed_dim (usually 1024 or 512)

    def forward(self, x):
        # x: embedding tensor (batch_size by seq_len by vocab_size)
        return torch.matmul(x, self.embed_matrix)
