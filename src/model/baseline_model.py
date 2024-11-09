from torch import nn
from torch.nn import Sequential

class SourceSeparator(nn.Module):
    def __init__(self, input_channels=2, hidden_channels=64, num_layers=3):
        super(SourceSeparator, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
        input_size=input_channels, 
        hidden_size=hidden_channels, 
        num_layers=num_layers, 
        batch_first=True,
        bidirectional=True
        )

        self.decoder = nn.LSTM(
        input_size=hidden_channels * 2,
        hidden_size=hidden_channels,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=True
        )

        self.output_layer = nn.Linear(hidden_channels * 2, input_channels) 

    def forward(self, mix_audio):
        _, (hidden_state, _) = self.encoder(mix_audio)
        hidden_state = hidden_state.view(self.num_layers, 2, self.hidden_channels)
        hidden_state = hidden_state[-1, :, :]
        
        decoded, _ = self.decoder(hidden_state.unsqueeze(0).repeat(mix_audio.shape[0], 1, 1)) 

        separated_sources = self.output_layer(decoded)
        
        return separated_sources
   
    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info