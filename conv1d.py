import torch


def conv1d(input: torch.Tensor,
           weight: torch.Tensor,
           bias: torch.Tensor = None):
    
    n, c_in, l_in = input.shape
    
    c_out, c_in_weight, kernel_size = weight.shape
    if c_in != c_in_weight:
        raise RuntimeError(f"Given input of size {list(input.shape)}, expected weight to be of "
                           f"size [*, {c_in}, *], but got size {weight.shape} instead.")
    
    c_out_bias, = bias.shape
    if c_out != c_out_bias:
        raise RuntimeError(f"Given input of size {list(input.shape)}, expected weight to be of "
                           f"size [*, {c_in}, *], but got size {weight.shape} instead.")

    l_out = l_in - kernel_size + 1

    indices_l_out = torch.arange(0, l_out).view(1, -1)
    indices_kernel_size = torch.arange(0, kernel_size).view(-1, 1)
    indices_of_sliding_window = indices_l_out + indices_kernel_size

    input_after_sliding_window = input[:, :, indices_of_sliding_window]
    return (weight.transpose(0, 1) @ input_after_sliding_window).sum(axis=1) + bias.view(1, -1, 1)






