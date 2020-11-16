import torch


def fft2(data, real=False):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        :param data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -2 & -1 are spatial dimensions and dimension -3 has size 1 or 2. All other dimensions are
            assumed to be batch dimensions -> Typically size B x 1 or 2 x H x W
        :param real
    Returns:
        :return torch.Tensor: The 2D FFT of the input, with dimension B x 2 x H x W
    """
    data = data.repeat(1, 2, 1, 1)
    data[:, 1] *= 0
    data = ifftshift(data, dim=(-2, -1))
    if not real:
        assert data.size(-3) == 2
        data = torch.fft(ctoend(data), 2, normalized=True)
    else:
        assert data.size(-3) == 1
        data = torch.rfft(data, 2, normalized=True, onesided=False).squeeze(-4)
    data = fftshift(data, dim=(-3, -2))
    return cto1(data)


def ifft2(data, real=False):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        :param data (torch.Tensor): Complex valued input data with at least 3 dimensions, typically of dimension
            B x 2 x H x W
        :param real
    Returns:
        :return torch.Tensor: The IFFT of the input, with dimension B x 1 or 2 x H x W
    """
    assert data.size(-3) == 2, data.shape
    data = ctoend(data)
    data = ifftshift(data, dim=(-3, -2))
    if not real:
        data = torch.ifft(data, 2, normalized=True)
        data = cto1(data)
    else:
        data = torch.irfft(data, 2, normalized=True, onesided=False).unsqueeze(-3)
    data = fftshift(data, dim=(-2, -1))
    return data[:, :1]


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    :param x: pt.Tenser to be shifted
    :param dim: dimensions to shift along
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def cto1(x):
    """
    Args:
        :param x (tensor) with three last dimensions H x W x (1 or 2)
    Returns:
        :return tensor with three last dimensions (1 or 2) x H x W
    """
    return x.transpose(-3, -1).transpose(-1, -2)

def ctoend(x):
    """
    Args:
        :param x (tensor) with three last dimensions (1 or 2) x H x W
    Returns:
        :return tensor with three last dimensions H x W x (1 or 2)
    """
    return x.transpose(-3, -1).transpose(-3, -2)

def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    :param x: pt.Tenser to be shifted
    :param dim: dimensions to shift along
    :return:
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    :param x: pt.Tenser to be shifted
    :param dim: dimensions to shift along
    :return:
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)
