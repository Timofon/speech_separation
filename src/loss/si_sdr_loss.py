import torch
from torch import nn

from src.metrics.utils import calc_si_sdri

class SI_SDRILoss(nn.Module):
    def forward(self, s1_audio: torch.Tensor, s2_audio: torch.Tensor, 
                s1_predicted: torch.Tensor, s2_predicted: torch.Tensor, mix_audio: torch.Tensor, **batch):
        s1_true = calc_si_sdri(s1_predicted, s1_audio, mix_audio)
        s2_true = calc_si_sdri(s2_predicted, s2_audio, mix_audio)

        s1_permuted = calc_si_sdri(s1_predicted, s2_audio, mix_audio)
        s2_permuted = calc_si_sdri(s2_predicted, s1_audio, mix_audio)

        loss = torch.maximum((s1_true + s2_true) / 2, (s1_permuted + s2_permuted) / 2)
        return {"loss": -torch.mean(loss)}    
    