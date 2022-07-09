from torch import nn

from multi_head_attention import MultiHeadAttention
from feedforward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, input_dim, num_heads):
        """
        embed_dim: num of expected features in input (same as d_model)
        input_dim: length of sequence
        num_heads: num of heads
        """
        super().__init__()

        self.attention = MultiHeadAttention(num_heads=num_heads, embed_dim=embed_dim, input_dim=input_dim, dropout=0.1)
        self.feedforward = PositionwiseFeedForward(embed_dim=embed_dim, input_dim=input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # attention output
        attn_out = self.attention(q=x, k=x, v=x, mask=mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # feedforward output
        ff_out = self.feedforward(x)  # TODO: needs to be implemented
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)

        return x