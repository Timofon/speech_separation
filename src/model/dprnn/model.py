import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model.dprnn.encoder import Encoder
from src.model.dprnn.decoder import Decoder
from src.model.dprnn.dprnn_block import DPRNNBlock

class DPRNNModel(nn.Module):
    def __init__(self, N=512, K=44, rnn_layers_num=6, rnn_hidden_size=128):
        super(DPRNNModel, self).__init__()
        
        self.N = N  # feature_num
        self.K = K  # chunk size
        self.P = K // 2  # hop size

        self.encoder = Encoder(encoder_num_features=N)
        
        self.dprnn_layers = nn.ModuleList(
            [
                DPRNNBlock(
                    input_size=N, hidden_size=rnn_hidden_size
                )
                for _ in range(rnn_layers_num)
            ]
        )

        self.masking_block = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(N, 2 * N, kernel_size=1),
            nn.Sigmoid()
        )

        self.decoder = Decoder()
    
    def forward(self, mix_audio: Tensor, **batch):
        '''
            x: [B, T]
            out: [B, 2, T]
        '''

        enc_x = self.encoder(mix_audio) # -> [B, N, L]

        B, N, L = enc_x.size()

        # separation
        x = self._segmentation(enc_x) # -> [B, N, K, S]
        for dprnn_block in self.dprnn_layers:
            x = dprnn_block(x) # -> [B, N, K, S]

        x = self._overlap_add(x, L) # -> [B, N, L]
        speaker_masks = self.masking_block(x) # -> [B, 2 * N, L]

        speaker_masks = speaker_masks.reshape(B, 2, N, L)

        out = speaker_masks * enc_x.unsqueeze(1)
        out = out.reshape(2 * B, N, L)
        decoded_outs = self.decoder(out) # -> [B, 2, L]

        return {"s1_predicted": decoded_outs[:, 0, :],
                "s2_predicted": decoded_outs[:, 1, :]}
    

    def _segmentation(self, x: Tensor):
        '''
            W: [B, N, L]
            out: [B, N, K, S]
        '''

        B, N, L = x.size()
        x = F.unfold(x.unsqueeze(-1), kernel_size=(self.K, 1), padding=(self.P, 0), stride=(self.P, 1))
        return x.reshape(B, N, self.K, -1)

    def _overlap_add(self, x: Tensor, L):
        '''
            x: [2 * B, N, K, S]
            out: 
        '''
        
        B, N, K, S = x.size()
        x = x.reshape(B, N * K, S)
        x = F.fold(input=x, output_size=(L, 1), kernel_size=(K, 1), padding=(self.P, 0), stride=(self.P, 1))
        # -> [2 * B, N, L]
        return x.squeeze(-1)
    
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
