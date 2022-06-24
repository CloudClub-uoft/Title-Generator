from torch import nn
import torch

from multi_head_attention import MultiHeadAttention
from feedforward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, input_dim, num_heads):
        """
        embed_dim: num of expected features in input (same as d_model)
        input_dim: length of sequence
        num_heads: num of heads
        """
        super().__init__()

        self.attention1 = MultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim, input_dim=input_dim, dropout=0.1)
        self.attention2 = MultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim, input_dim=input_dim, dropout=0.1)
        self.feedforward = PositionwiseFeedForward(embed_dim=embed_dim, input_dim=input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # masked attention output
        attn_1_out = self.attention1(q=x, k=x, v=x, mask=mask)
        x = x + self.dropout1(attn_1_out)
        x = self.norm1(x)

        # unmasked attention output
        attn_2_out = self.attention2(q=x, k=x, v=x, mask=None)
        x = x + self.dropout2(attn_2_out)
        x = self.norm2(x)

        # feedforward output
        ff_out = self.feedforward(x)  # TODO: needs to be implemented
        x = x + self.dropout3(ff_out)
        x = self.norm3(x)

        return x

if __name__ == "__main__":
    # TESTING
    embed_dim = 3
    num_heads = 1

    x = torch.tensor([[0, 10, 0]], dtype=torch.float32)
    input_dim = 3

    decoder = DecoderLayer(embed_dim, input_dim, num_heads)
    output = decoder.forward(x)
    print(output)