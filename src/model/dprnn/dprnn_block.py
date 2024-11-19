import torch.nn as nn
from torch import Tensor
from torch.nn import LSTM

class DPRNNBlock(nn.Module):
    def __init__(self, input_size, hidden_size=128) -> None:
        super(DPRNNBlock, self).__init__()

        self.intra_chunk_rnn = LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.intra_linear = nn.Linear(in_features=hidden_size * 2, out_features=input_size)
        self.intra_layer_norm = nn.GroupNorm(num_groups=1, num_channels=input_size)

        self.inter_chunk_rnn = LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.inter_linear = nn.Linear(in_features=hidden_size * 2, out_features=input_size)
        self.inter_layer_norm = nn.GroupNorm(num_groups=1, num_channels=input_size)

    def forward(self, x: Tensor):
        '''
            x: [B, N, K, S]
            out: [B, N, K, S]
        '''

        B, N, K, S = x.size()
        x_skip = x
        
        # intra-chunk processing
        x = x.permute(0, 3, 2, 1).reshape(B * S, K, N)
        x, _ = self.intra_chunk_rnn(x)
        x = self.intra_linear(x)
        x = x.reshape(B, S, K, N).permute(0, 3, 2, 1)
        x = self.intra_layer_norm(x)

        x = x + x_skip
        x_skip = x

        # inter-chunk processing
        x = x.permute(0, 2, 3, 1).reshape(B * K, S, N)
        x, _ = self.inter_chunk_rnn(x)
        x = self.inter_linear(x)
        x = x.reshape(B, K, S, N).permute(0, 3, 1, 2)
        x = self.inter_layer_norm(x)

        out = x + x_skip
        return out
