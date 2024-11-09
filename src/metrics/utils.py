from torchmetrics.audio import SignalDistortionRatio


def calc_si_sdri(predicted, target, mix):
    si_sdr = SignalDistortionRatio()

    return si_sdr(predicted, target) - si_sdr(mix, target)
