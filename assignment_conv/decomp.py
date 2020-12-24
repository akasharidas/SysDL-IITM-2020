import torch
from decomposition.tucker import tucker_decomp
from decomposition.cp import cp_decomp


class Tucker_Conv2d(torch.nn.Module):
    def __init__(self, C, M, R, S, _filter):
        super().__init__()
        layer = torch.nn.Conv2d(
            in_channels=C, out_channels=M, kernel_size=(R, S), bias=False
        )
        layer.weight = torch.nn.Parameter(_filter.to("cpu"))
        self.layer = tucker_decomp(layer)

    def forward(self, _input, _filter=None):
        return self.layer(_input)


class CP_Conv2d(torch.nn.Module):
    def __init__(self, C, M, R, S, _filter):
        super().__init__()
        layer = torch.nn.Conv2d(
            in_channels=C, out_channels=M, kernel_size=(R, S), bias=False
        )
        layer.weight = torch.nn.Parameter(_filter.to("cpu"))
        self.layer = cp_decomp(layer)

    def forward(self, _input, _filter=None):
        return self.layer(_input)
