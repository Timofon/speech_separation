from torchmetrics.audio import ScaleInvariantSignalDistortionRatio, ScaleInvariantSignalNoiseRatio


def calc_si_sdr(predicted, target):
    si_sdr = ScaleInvariantSignalDistortionRatio().to('cuda')
    
    return si_sdr(predicted, target)


def calc_si_sdri(predicted, target, mix):
    si_sdr = ScaleInvariantSignalDistortionRatio().to('cuda')

    return si_sdr(predicted, target) - si_sdr(mix, target)

def calc_si_snr(predicted, target):
    si_snr = ScaleInvariantSignalNoiseRatio().to('cuda')
    
    return si_snr(predicted, target)


def calc_si_snri(predicted, target, mix):
    si_snr = ScaleInvariantSignalNoiseRatio().to('cuda')

    return si_snr(predicted, target) - si_snr(mix, target)