from torch import nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):

    def __init__(self, embed_dim, input_dim, dropout_rate=0.1):
        """
        embed_dim: num of expected features in input (same as d_model)
        input_dim: length of sequence
        """
        super(PositionwiseFeedForward, self).__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.w_1 = nn.Linear(embed_dim, input_dim)
        self.w_2 = nn.Linear(input_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x = (batch_size, input_dim, embed_dim)
        x = self.dropout(F.relu(self.w_1(x))) 
        x = self.w_2(x)  
        return x