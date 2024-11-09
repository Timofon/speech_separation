import torch
from torch import nn

from src.metrics.utils import calc_si_sdri

class SI_SDRILoss(nn.Module):
    def forward(self, s1: torch.Tensor, s2: torch.Tensor, 
                s1_pred: torch.Tensor, s2_pred: torch.Tensor, mix: torch.Tensor, **batch):
        s1_s1 = calc_si_sdri(s1, s1_pred, mix)
        s1_s2 = calc_si_sdri(s1, s2_pred, mix)
        s2_s1 = calc_si_sdri(s2, s1_pred, mix)
        s2_s2 = calc_si_sdri(s2, s2_pred, mix)

        loss = torch.maximum((s1_s1 + s2_s2) / 2, (s1_s2 + s2_s1) / 2)
        return {"loss": -torch.mean(loss)}
