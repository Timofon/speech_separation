import torch
from torch import nn
from src.model.convtasnet.conv_block import ConvBlock


class Masker(nn.Module):
    def __init__(self, original_dim, num_speakers, kernel_size, num_features, hidden_size, num_blocks, num_modules):
        super().__init__()

        self.original_dim = original_dim
        self.num_speakers = num_speakers

        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=original_dim)
        self.conv_expanding = nn.Conv1d(in_channels=original_dim, out_channels=num_features, kernel_size=1)
        self.conv_compressing = nn.Conv1d(in_channels=num_features, out_channels=original_dim*num_speakers, kernel_size=1)

        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

        def _is_last(block_index, module_index, num_blocks, num_modules):
            is_last_layer_in_module = (block_index == (num_blocks - 1))
            is_last_module = (module_index == (num_modules - 1))
            return is_last_layer_in_module and is_last_module

        convs = []
        for module in range(num_modules):
            for block in range(num_blocks):
                factor = 2 ** block

                convs.append(ConvBlock(original_num_channels=num_features,
                                       hidden_size=hidden_size,
                                       kernel_size=kernel_size,
                                       dilation=factor,
                                       padding=factor,
                                       is_last=_is_last(block_index=block,
                                                        module_index=module,
                                                        num_blocks=num_blocks,
                                                        num_modules=num_modules)))
        
        self.convs = nn.ModuleList(convs)
    
    def forward(self, latent):
        latent = self.group_norm(latent)
        latent = self.conv_expanding(latent)

        skip_connections = []

        for layer in self.convs:
            residual, skip = layer(latent)
            if residual is not None:
                latent = latent + residual
            skip_connections.append(skip)
        
        latent = torch.sum(torch.stack(skip_connections), dim=0)
    
        latent = self.prelu(latent)
        latent = self.conv_compressing(latent)
        latent = self.sigmoid(latent)
        return latent.reshape(latent.shape[0], self.num_speakers, self.original_dim, -1)
