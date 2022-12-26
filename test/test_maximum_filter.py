from scipy.ndimage import maximum_filter as maximum_filter_scipy
import numpy as np
import torch
from superflux import maximum_filter

def test_maximum_filter():
    array = np.random.rand(64, 32)
    scipy_out = maximum_filter_scipy(array, 3, mode='constant')
    tensor = torch.tensor(array)
    tensor = tensor[None, :, :]
    torch_out = maximum_filter(tensor, 3)
    assert scipy_out[None, :, :].shape == torch_out.shape
    assert (torch.tensor(scipy_out) == torch_out).all

def test_maximum_filter_rectangle_kernel():
    array = np.random.rand(64, 32)
    scipy_out = maximum_filter_scipy(array, [1, 3], mode='constant')
    tensor = torch.tensor(array)
    tensor = tensor[None, :, :]
    torch_out = maximum_filter(tensor, [1, 3])
    assert scipy_out[None, :, :].shape == torch_out.shape
    assert (torch.tensor(scipy_out) == torch_out).all


def test_maximum_filter_reflect_mode():
    array = np.random.rand(64, 32)
    scipy_out = maximum_filter_scipy(array, 3, mode='reflect')
    tensor = torch.tensor(array)
    tensor = tensor[None, :, :]
    torch_out = maximum_filter(tensor, 3, mode='reflect')
    assert scipy_out[None, :, :].shape == torch_out.shape
    assert (torch.tensor(scipy_out) == torch_out).all


def test_maximum_filter_reflect_mode_rectangle_kernel():
    array = np.random.rand(64, 32)
    scipy_out = maximum_filter_scipy(array, [1, 3], mode='reflect')
    tensor = torch.tensor(array)
    tensor = tensor[None, :, :]
    torch_out = maximum_filter(tensor, [1, 3], mode='reflect')
    assert scipy_out[None, :, :].shape == torch_out.shape
    assert (torch.tensor(scipy_out) == torch_out).all
