import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Decoder(nn.Module):
    def __init__(self, decoder_in_features=512, decoder_kernel_size=64) -> None:
        super(Decoder, self).__init__()

        self.decoder_conv = nn.ConvTranspose1d(
            in_channels=decoder_in_features,
            out_channels=1,
            kernel_size=decoder_kernel_size,
            stride=decoder_kernel_size // 2,
            padding=decoder_kernel_size // 2
        )

    def forward(self, x: Tensor):
        '''
            x: [2 * B, N, L]
            out: [B, 2, L]
        '''

        B, _, _ = x.size()
        return self.decoder_conv(x).reshape(B // 2, 2, -1)