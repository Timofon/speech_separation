from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, original_num_channels, hidden_size, kernel_size, padding, dilation, is_last):
        super().__init__()

        self.conv = nn.Conv1d(in_channels=original_num_channels, out_channels=hidden_size, kernel_size=1)
        self.d_conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_size)
        self.residual = nn.Conv1d(in_channels=hidden_size, out_channels=original_num_channels, kernel_size=1)
        self.skip_con = nn.Conv1d(in_channels=hidden_size, out_channels=original_num_channels, kernel_size=1)

        self.group_norm1 = nn.GroupNorm(num_groups=1, num_channels=hidden_size)
        self.group_norm2 = nn.GroupNorm(num_groups=1, num_channels=hidden_size)
        
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

        self.is_last = is_last
    
    def forward(self, latent):
        latent = self.conv(latent)
        latent = self.prelu1(latent)
        latent = self.group_norm1(latent)
        latent = self.d_conv(latent)
        latent = self.prelu2(latent)
        latent = self.group_norm2(latent)
        
        return self.residual(latent) if not self.is_last else None, self.skip_con(latent)
