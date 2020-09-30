import torch
import torch.nn as nn
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class GaussianNoise(nn.Module):
    """
    Create Gaussian noise on the input with specified signal to noise ration snr.
    """
    def __init__(self, snr):
        super(GaussianNoise, self).__init__()
        self.snr = snr

    def forward(self, y):
        std = torch.std(y, dim=1) * np.power(10.0, -self.snr / 20)
        noise = torch.normal(torch.zeros_like(y, device=device),
                             std=(torch.zeros_like(y, device=device) + std.reshape(-1, 1)))
        return y + noise
