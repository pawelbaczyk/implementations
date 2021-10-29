import torch
import math
import torch.nn.functional as F


def _test_tensor_shapes_for_conv(input: torch.Tensor,
                                 weight: torch.Tensor,
                                 bias: torch.Tensor = None):
    c_in = input.shape[1]

    c_out = weight.shape[0]
    c_in_weight = weight.shape[1]
    if c_in != c_in_weight:
        raise RuntimeError(f"Given input of size {list(input.shape)}, expected weight to be of "
                           f"size [..., {c_in}, ...], but got size {list(weight.shape)} instead.")

    if bias is not None:
        c_out_bias, = bias.shape
        if c_out != c_out_bias:
            raise RuntimeError(f"Given weight of size {list(weight.shape)}, expected bias to be of"
                               f" size [{c_out}], but got size {list(bias.shape)} instead.")


def conv2d(input: torch.Tensor,
           weight: torch.Tensor,
           bias: torch.Tensor = None,
           stride: int = 1,
           padding: int = 0):

    _test_tensor_shapes_for_conv(input, weight, bias)

    c_out = weight.shape[0]
    h_k = weight.shape[2]

    shape_out = torch.floor(
        (torch.tensor(input.shape[2:]) + 2 * padding - torch.tensor(weight.shape[2:])) / stride + 1
    ).int().tolist()

    if bias is None:
        bias = torch.zeros(c_out)

    input = F.pad(input, (padding, padding) * (len(input.shape) - 2))

    h_out = shape_out[0]
    if len(input.shape) > 3:
        w_in = input.shape[3]
        w_k = weight.shape[3]
        w_out = shape_out[1]
    else:
        w_in = w_k = w_out = 1

    i1 = torch.arange(w_k).repeat(h_k) + torch.repeat_interleave(torch.arange(0, w_in * h_k, w_in), w_k)
    i2 = torch.arange(w_out).repeat(h_out) + torch.repeat_interleave(torch.arange(0, w_in * h_out, w_in), w_out)

    i3 = i1.view(-1, 1) + i2.view(1, -1)

    input = torch.flatten(input, 2)[:, :, i3]

    return ((torch.flatten(weight, 2).transpose(0, 1) @ input).sum(axis=1) + bias.view(1, -1, 1)).unflatten(-1, shape_out)

input = torch.tensor([[[[0,1,2,3,4,5],[6,7,8,9,10,11],[12,13,14,15,16,17],[18,19,20,21,22,23]]]])
weight = torch.tensor([[[[1,1,1],[1,1,1]]]])

c = conv2d(input, weight)
print(c)

input = torch.tensor([[[3, 2, 1, 5, 6, 7]], [[2, 3, 4, 0, -1, 0]]])
weight = torch.tensor([[[1, 2, 3]]])

c = conv2d(input, weight)
print(c)



