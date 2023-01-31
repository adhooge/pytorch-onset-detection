import superflux2 as sf
import numpy as np
import torch


def test_wavRead():
    wav = sf.Wav("test/files/clap.wav")
    arr = np.loadtxt("test/files/read_audio.txt", dtype=np.int16)
    arr_tensor = torch.from_numpy(arr)
    wav.downmix()
    assert torch.allclose(wav.audio.nonzero(), arr_tensor.nonzero())
    assert torch.equal(wav.audio, arr_tensor)


def test_frequencies():
    expected = torch.Tensor([220, 311.12698, 440, 622.253967, 880])
    frequencies = sf.Filter.frequencies(fmin=220, fmax=880, bands=2, ref_freq=440)
    assert torch.allclose(expected, frequencies)
