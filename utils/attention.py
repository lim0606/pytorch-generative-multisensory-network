import torch


def pack_tensor_list(tensor_list, batch_sizes, pad=None):
    # init
    max_batch_size = max(batch_sizes)
    sizes = list(tensor_list[0].size())
    sizes[0] = 1

    # init pad_tensor
    if pad is None:
        pad = tensor_list[0].new_zeros(sizes)

    # pack
    packed_tensor = []
    for i, tensor in enumerate(tensor_list):
        # init
        padded_tensor = [tensor]

        # add pad representation
        pad_size = max_batch_size - batch_sizes[i]
        if pad_size > 0:
            sizes[0] = pad_size
            pad_tensor = pad.expand(sizes)
            padded_tensor += [pad_tensor]
        else:
            pad_tensor = None
            padded_tensor += []

        # concat
        padded_tensor = torch.cat(padded_tensor, dim=0)

        # append to packed_tensor
        packed_tensor += [padded_tensor.unsqueeze(0)]

    # concat packed_tensor
    packed_tensor = torch.cat(packed_tensor, dim=0)

    return packed_tensor


def flatten_packed_tensor(packed_tensor, batch_sizes):
    # init
    max_batch_size = max(batch_sizes)

    # chunk
    packed_tensor_list = torch.chunk(packed_tensor, len(batch_sizes))

    # split
    flatten_tensor = []
    for i in range(len(batch_sizes)):
        batch_size = batch_sizes[i]
        flatten_tensor += [packed_tensor_list[i].squeeze(0)[:batch_size]]

    # concat
    flatten_tensor = torch.cat(flatten_tensor, dim=0)

    return flatten_tensor
