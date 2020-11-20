import torch
import torch.nn as nn
import numpy as np

import utils.conf as conf

device = conf.device


class GaussianNoise(nn.Module):
    """
    Create Gaussian noise on the input with specified signal to noise ration snr.
    """

    def __init__(self, snr, comp=False):
        super(GaussianNoise, self).__init__()
        self.snr = snr
        self.comp = comp

    def forward(self, y):
        y_ = y.reshape(y.shape[0], -1)

        if self.comp:
            # std nicht instantan...
            std_y_bar = torch.sqrt((torch.std(y[:, :, 0], dim=-1) ** 2 + torch.std(y[:, :, 1], dim=-1) ** 2).mean())
            std = std_y_bar * np.power(10.0, -self.snr / 20)
            c_std = std * 1.0 / np.sqrt(2)
            noise = torch.normal(torch.zeros_like(y_, device=device),
                                 std=(torch.zeros_like(y_, device=device) + c_std.reshape(-1, 1)))
            print('noise')
            print(std)
            print((torch.norm(noise[:, :, 0], dim=-1) ** 2 + torch.norm(noise[:, :, 1], dim=-1) ** 2).mean() / 1024)
        else:
            std = torch.std(y_, dim=1) * np.power(10.0, -self.snr / 20)
            noise = torch.normal(torch.zeros_like(y_, device=device),
                                 std=(torch.zeros_like(y_, device=device) + std.reshape(-1, 1)))
        return y + noise.reshape(y.shape)
