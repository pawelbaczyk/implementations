import torch
import math
import torch.nn.functional as F


def _test_tensor_shapes_for_conv1d(input: torch.Tensor,
                                   weight: torch.Tensor,
                                   bias: torch.Tensor = None):
    n, c_in, l_in = input.shape

    c_out, c_in_weight, kernel_size = weight.shape
    if c_in != c_in_weight:
        raise RuntimeError(f"Given input of size {list(input.shape)}, expected weight to be of "
                           f"size [*, {c_in}, *], but got size {list(weight.shape)} instead.")

    if bias is not None:
        c_out_bias, = bias.shape
        if c_out != c_out_bias:
            raise RuntimeError(f"Given weight of size {list(weight.shape)}, expected bias to be of"
                               f" size [{c_out}], but got size {list(bias.shape)} instead.")


def _expand_with_sliding_window(input, l_out, kernel_size, stride):
    indices_l_out = stride * torch.arange(0, l_out).view(1, -1)
    indices_kernel_size = torch.arange(0, kernel_size).view(-1, 1)
    indices_of_sliding_window = indices_l_out + indices_kernel_size

    return input[:, :, indices_of_sliding_window]


def conv1d(input: torch.Tensor,
           weight: torch.Tensor,
           bias: torch.Tensor = None,
           stride: int = 1,
           padding: int = 0):

    _test_tensor_shapes_for_conv1d(input, weight, bias)

    c_out = weight.shape[0]
    l_in = input.shape[2]
    kernel_size = weight.shape[2]

    l_out = math.floor((l_in + 2 * padding - kernel_size) / stride + 1)

    if bias is None:
        bias = torch.zeros(c_out)

    input = F.pad(input, (padding, padding))
    input = _expand_with_sliding_window(input, l_out, kernel_size, stride)

    return (weight.transpose(0, 1) @ input).sum(axis=1) + bias.view(1, -1, 1)






