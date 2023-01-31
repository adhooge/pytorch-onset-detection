import torch
import torchaudio
from math import floor, ceil, log

class Wav:
    """
    Wrapper class around torchaudio.load.
    """
    def __init__(self, filename):
        self.audio, self.rate = torchaudio.load(filename, normalize=False, channels_first=False)
        # self.audio shape is [samples, channel]
        self.num_samples = self.audio.shape[0]
        self.duration = self.num_samples / self.rate
        self.num_channels = self.audio.shape[1]

    def downmix(self):
        if self.num_channels > 1:
            # I must cast to a floating point dtype to use mean
            self.audio = torch.mean(self.audio, dim=-1, dtype=torch.double)
        else:
            self.audio = self.audio[:, 0]

    def normalize(self):
        self.audio = self.audio / torch.max(torch.abs(self.audio))


class Filter:
    def __init__(self, num_fft_bins, rate, bands: int = 24, fmin: int = 30,
                fmax: int = 17000, equal: bool = False) -> None:
        self.rate = rate
        if fmax > rate / 2:
            # ever heard about Shannon?
            fmax = rate / 2
        frequencies = self.frequencies(bands, fmin, fmax)

    @staticmethod
    def frequencies(bands, fmin, fmax, ref_freq: int = 440):
        factor = 2**(1/bands)
        lower_pow = floor((log(fmin) - log(ref_freq))/log(factor))
        upper_pow = ceil((log(fmax) - log(ref_freq))/log(factor))
        # Add an epsilon to include last step
        powers = torch.arange(lower_pow, upper_pow + 1e-6, step = 1)
        factors = torch.pow(factor, powers)
        frequencies = ref_freq * factors
        return frequencies
