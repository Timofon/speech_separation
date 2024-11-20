import torch
from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_si_sdri


class SISDRiMetric(BaseMetric):
    def __call__(self, s1_audio: torch.Tensor, s2_audio: torch.Tensor, 
                 s1_predicted: torch.Tensor, s2_predicted: torch.Tensor, mix_audio: torch.Tensor, **batch):
        s1_true = calc_si_sdri(s1_predicted, s1_audio, mix_audio)
        s2_true = calc_si_sdri(s2_predicted, s2_audio, mix_audio)

        s1_permuted = calc_si_sdri(s1_predicted, s2_audio, mix_audio)
        s2_permuted = calc_si_sdri(s2_predicted, s1_audio, mix_audio)

        return torch.mean(torch.maximum((s1_true + s2_true) / 2, (s1_permuted + s2_permuted) / 2))