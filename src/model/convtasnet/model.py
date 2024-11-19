from torch import nn
from src.model.convtasnet.masker import Masker


class ConvTasNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_speakers = 2
        self.encoder_num_features = 512
        self.masker_num_features = 128
        self.hidden_size = 512
        self.stride = 8
        self.blocks = 8
        self.num_modules = 3

        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=self.encoder_num_features,
            kernel_size=16,
            stride=self.stride,
            padding=self.stride
        )

        self.masker = Masker(
            original_dim=self.encoder_num_features,
            num_speakers=self.num_speakers,
            kernel_size=3,
            num_features=self.masker_num_features,
            hidden_size=self.hidden_size,
            num_blocks=self.blocks,
            num_modules=self.num_modules
        )

        self.decoder = nn.ConvTranspose1d(
            in_channels=self.encoder_num_features,
            out_channels=1,
            kernel_size=16,
            stride=self.stride,
            padding=self.stride
        )
        
    def forward(self, mix_audio, **batch):
        mix_audio = mix_audio.unsqueeze(1)
    
        latents = self.encoder(mix_audio)

        masks = self.masker(latents)
        masked_latents = masks * latents.unsqueeze(1)

        masked_latents = masked_latents.view(mix_audio.shape[0] * self.num_speakers, self.encoder_num_features, -1)
        
        decoded_latents = self.decoder(masked_latents)

        output = decoded_latents.view(mix_audio.shape[0], self.num_speakers, mix_audio.shape[2])

        return {"s1_predicted": output[:, 0, :],
                "s2_predicted": output[:, 1, :]}

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
