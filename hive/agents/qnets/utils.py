import torch


def calculate_output_dim(net, input_dim):
    if isinstance(input_dim, int):
        input_dim = tuple(
            input_dim,
        )
    placeholder = torch.zeros((0,) + tuple(input_dim))
    output = net(placeholder)
    return output.size()[1:]
