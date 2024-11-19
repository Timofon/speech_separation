import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, encoder_num_features=512, encoder_kernel_size=64) -> None:
        super(Encoder, self).__init__()

        self.encoder_conv = nn.Conv1d(
            in_channels=1,
            out_channels=encoder_num_features, # N
            kernel_size=encoder_kernel_size,
            stride=encoder_kernel_size // 2,
            padding=encoder_kernel_size // 2
        )
    
    def forward(self, x: Tensor):
        '''
            x: [B, T]
            out: [B, N, L]
        '''

        x = torch.unsqueeze(x, dim=1)
        return self.encoder_conv(x)
