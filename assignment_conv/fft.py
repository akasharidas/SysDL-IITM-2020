"""
Modified from: https://github.com/fkodom/fft-conv-pytorch
"""


from functools import partial
from typing import Tuple, Union, Iterable

import torch
from torch import nn, Tensor
from torch.fft import rfftn, irfftn
import torch.nn.functional as f


class FFT_Conv2d(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def to(self, device):
        super().to(device)
        self.device = device

    @staticmethod
    def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
        scalar_matmul = partial(torch.einsum, "agc..., gbc... -> agb...")
        a = a.view(a.size(0), groups, -1, *a.shape[2:])
        b = b.view(groups, -1, *b.shape[1:])

        real = scalar_matmul(a.real, b.real) - scalar_matmul(a.imag, b.imag)
        imag = scalar_matmul(a.imag, b.real) + scalar_matmul(a.real, b.imag)
        c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
        c.real, c.imag = real, imag

        return c.view(c.size(0), -1, *c.shape[3:])

    @staticmethod
    def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
        if isinstance(val, Iterable):
            out = tuple(val)
            if len(out) == n:
                return out
            else:
                raise ValueError(
                    f"Cannot cast tuple of length {len(out)} to length {n}."
                )
        else:
            return n * (val,)

    def forward(
        self,
        signal: Tensor,
        kernel: Tensor,
        bias: Tensor = None,
        padding: Union[int, Iterable[int]] = 0,
        stride: Union[int, Iterable[int]] = 1,
        groups: int = 1,
    ) -> Tensor:
        # Cast padding & stride to tuples.
        padding_ = self.to_ntuple(padding, n=signal.ndim - 2)
        stride_ = self.to_ntuple(stride, n=signal.ndim - 2)

        # Pad the input signal & kernel tensors
        signal_padding = [p for p in padding_[::-1] for _ in range(2)]
        signal = f.pad(signal, signal_padding)

        # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
        # have *even* length.  Just pad with one more zero if the final dimension is odd.
        if signal.size(-1) % 2 != 0:
            signal_ = f.pad(signal, [0, 1])
        else:
            signal_ = signal

        kernel_padding = [
            pad
            for i in reversed(range(2, signal_.ndim))
            for pad in [0, signal_.size(i) - kernel.size(i)]
        ]
        padded_kernel = f.pad(kernel, kernel_padding)

        # Perform fourier convolution -- FFT, matrix multiply, then IFFT
        # signal_ = signal_.reshape(signal_.size(0), groups, -1, *signal_.shape[2:])
        signal_fr = rfftn(signal_, dim=tuple(range(2, signal.ndim)))
        kernel_fr = rfftn(padded_kernel, dim=tuple(range(2, signal.ndim)))

        kernel_fr.imag *= -1
        output_fr = self.complex_matmul(signal_fr, kernel_fr, groups=groups)
        output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))

        # Remove extra padded values
        crop_slices = [slice(0, output.size(0)), slice(0, output.size(1))] + [
            slice(0, (signal.size(i) - kernel.size(i) + 1), stride_[i - 2])
            for i in range(2, signal.ndim)
        ]
        output = output[crop_slices].contiguous()

        # Optionally, add a bias term before returning.
        if bias is not None:
            bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
            output += bias.view(bias_shape)

        return output
