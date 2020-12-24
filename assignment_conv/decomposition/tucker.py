from torch import nn
import numpy as np
import tensorly as tl
from tensorly.decomposition import partial_tucker
from .TVBMF import EVBMF

tl.set_backend("pytorch")


def tucker_decomp(layer):
    W = layer.weight.data
    rank = tucker_rank(layer)

    core, [last, first] = partial_tucker(W, modes=[0, 1], rank=rank, init="svd")

    first_layer = nn.Conv2d(
        in_channels=first.shape[0],
        out_channels=first.shape[1],
        kernel_size=1,
        padding=0,
        bias=False,
    )

    core_layer = nn.Conv2d(
        in_channels=core.shape[1],
        out_channels=core.shape[0],
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        bias=False,
    )

    last_layer = nn.Conv2d(
        in_channels=last.shape[1],
        out_channels=last.shape[0],
        kernel_size=1,
        padding=0,
        bias=True,
    )

    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    fk = first.t_().unsqueeze_(-1).unsqueeze_(-1)
    lk = last.unsqueeze_(-1).unsqueeze_(-1)

    first_layer.weight.data = fk
    last_layer.weight.data = lk
    core_layer.weight.data = core

    new_layers = nn.Sequential(*[first_layer, core_layer, last_layer])

    return new_layers


def tucker_rank(layer):
    W = layer.weight.data
    mode3 = tl.base.unfold(W, 0)
    mode4 = tl.base.unfold(W, 1)
    diag_0 = EVBMF(mode3)
    diag_1 = EVBMF(mode4)
    d1 = diag_0.shape[0]
    d2 = diag_1.shape[1]

    # round to multiples of 16
    return [int(np.ceil(d1 / 16) * 16), int(np.ceil(d2 / 16) * 16)]
