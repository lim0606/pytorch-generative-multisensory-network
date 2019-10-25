import math

import torch
import torch.nn as nn
import torch.nn.functional as F


''' instance normalization for 1d input '''
def unit_norm(input):
    '''
    input = b x c
    output = b x c
    '''
    output = input / (torch.norm(input, dim=1, keepdim=True) + 1e-6)
    return output


''' instance normalization for 1d input '''
def instance_norm(input):
    '''
    input = b x c
    output = b x c
    '''
    output = F.instance_norm(input.unsqueeze(1)).squeeze(1)
    return output


''' laplacian filter (edge detection) for 1d input '''
# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    #gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
    #                            kernel_size=kernel_size, groups=channels, bias=False)

    #gaussian_filter.weight.data = gaussian_kernel
    #gaussian_filter.weight.requires_grad = False
    #return gaussian_filter
    return gaussian_kernel
#filt = get_gaussian_kernel()
#pad = nn.ReflectionPad1d(1)
GAUSSIAN_KERNEL = get_gaussian_kernel()

def laplacian_filter(input):
    global GAUSSIAN_KERNEL 
    if GAUSSIAN_KERNEL.device != input.device:
        GAUSSIAN_KERNEL = GAUSSIAN_KERNEL.to(input.device)
    batch_size = input.size(0)
    #output = net(input.unsqueeze(1)).squeeze(1)
    # pad
    #padded_input = pad(input.view(batch_size, 1, -1)).unsqueeze(2)
    padded_input = F.pad(input.view(batch_size, 1, -1), (1,1), mode='reflect').unsqueeze(2)
    buff = torch.cat([padded_input, padded_input, padded_input], dim=2)
    #output = filt(buff).view(batch_size, -1)
    output = F.conv2d(buff, GAUSSIAN_KERNEL, groups=1).view(batch_size, -1)
    return output


''' for haptic dataset '''
def _normalize(_train_data, method):
    # init
    train_data = _train_data.new_zeros(_train_data.size())
    num_dim = _train_data.size(1)

    # normalize
    ind_start = 0 # joint-pos
    ind_end = 22 # joint-pos
    train_data[:, ind_start:ind_end] = method(_train_data[:, ind_start:ind_end])

    ind_start = 22 # joint-vel
    ind_end = 44 # joint-vel
    train_data[:, ind_start:ind_end] = method(_train_data[:, ind_start:ind_end])

    ind_start = 44 # pos
    ind_end = 57 # pos
    train_data[:, ind_start:ind_end] = method(_train_data[:, ind_start:ind_end])

    ind_start = 57 # vel
    ind_end = 70 # vel
    train_data[:, ind_start:ind_end] = method(_train_data[:, ind_start:ind_end])

    ind_start = 70 # fc
    ind_end = 83 # fc
    train_data[:, ind_start:ind_end] = method(_train_data[:, ind_start:ind_end])

    ind_start = 83 # acc
    ind_end = 98 # acc
    train_data[:, ind_start:ind_end] = method(_train_data[:, ind_start:ind_end])

    ind_start = 98 # gyro
    ind_end = 113 # gyro
    train_data[:, ind_start:ind_end] = method(_train_data[:, ind_start:ind_end])

    if num_dim > 113:
        ind_start = 113 # touch
        ind_end = 132 # touch
        train_data[:, ind_start:ind_end] = method(_train_data[:, ind_start:ind_end])

    return train_data

def normalize(input, method=None):
    assert method is None or method in ['instance_norm', 'laplacian_filter']
    if method is None or input is None:
        output = input
    elif method == 'instance_norm':
        output = _normalize(input, instance_norm)
    elif method == 'laplacian_filter':
        output = _normalize(input, laplacian_filter)
    elif method == 'unit_norm':
        output = _normalize(input, unit_norm)
    else:
        raise NotImplementedError

    return output
