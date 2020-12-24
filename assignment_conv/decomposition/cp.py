import torch
import torch.nn as nn
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from .TVBMF import EVBMF

tl.set_backend("pytorch")


def cp_decomp(layer):
    W = layer.weight.data
    rank = est_rank(layer)

    _, temp = parafac(W, rank=rank, init="svd")
    last, first, vertical, horizontal = temp

    pointwise_s_to_r_layer = nn.Conv2d(
        in_channels=first.shape[0],
        out_channels=first.shape[1],
        kernel_size=1,
        padding=0,
        bias=False,
    )

    depthwise_r_to_r_layer = nn.Conv2d(
        in_channels=rank,
        out_channels=rank,
        kernel_size=vertical.shape[0],
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=rank,
        bias=False,
    )

    pointwise_r_to_t_layer = nn.Conv2d(
        in_channels=last.shape[1],
        out_channels=last.shape[0],
        kernel_size=1,
        padding=0,
        bias=True,
    )

    if layer.bias is not None:
        pointwise_r_to_t_layer.bias.data = layer.bias.data

    sr = first.t_().unsqueeze_(-1).unsqueeze_(-1)
    rt = last.unsqueeze_(-1).unsqueeze_(-1)
    rr = torch.stack(
        [
            vertical.narrow(1, i, 1) @ torch.t(horizontal).narrow(0, i, 1)
            for i in range(rank)
        ]
    ).unsqueeze_(1)

    pointwise_s_to_r_layer.weight.data = sr
    pointwise_r_to_t_layer.weight.data = rt
    depthwise_r_to_r_layer.weight.data = rr

    new_layers = nn.Sequential(
        *[pointwise_s_to_r_layer, depthwise_r_to_r_layer, pointwise_r_to_t_layer]
    )

    return new_layers


def est_rank(layer):
    W = layer.weight.data
    mode3 = tl.base.unfold(W, 0)
    mode4 = tl.base.unfold(W, 1)
    diag_0 = EVBMF(mode3)
    diag_1 = EVBMF(mode4)

    # round to multiples of 16
    return int(np.ceil(max([diag_0.shape[0], diag_1.shape[0]]) / 16) * 16)
