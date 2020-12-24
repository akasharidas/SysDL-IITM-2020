"""
Source: https://github.com/adam-dziedzic/winograd
"""

import torch


class Winograd_Conv2d(torch.nn.Module):
    def __init__(self, filter_value=None):
        super(Winograd_Conv2d, self).__init__()

        self.B = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
            ]
        )
        self.B_T = self.B.transpose(1, 0)
        self.G = torch.tensor(
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.0, 0.0, 1.0]]
        )
        self.G_T = self.G.transpose(1, 0)
        self.A = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0], [0.0, -1.0]])
        self.A_T = self.A.transpose(1, 0)

        if filter_value is not None:
            self.filter = filter_value

    def _apply(self, fn):
        super()._apply(fn)
        self.B = fn(self.B)
        self.B_T = fn(self.B_T)
        self.G = fn(self.G)
        self.G_T = fn(self.G_T)
        self.A = fn(self.A)
        self.A_T = fn(self.A_T)
        return self

    def forward(self, input, filter):
        """
        Compute Winograd convolution.
        :param input:
        :param filter:
        :return: output
        """
        N, C, H, W = input.size()
        K, Cprime, r, rprime = filter.size()
        assert H == W
        assert r == rprime
        assert C == Cprime
        m = 2
        a = m + r - 1
        overlap = r - 1

        if (H >= 4 and H % 2 == 0) is False:
            raise Exception("Only input for perfect tiling is supported.")

        input = torch.transpose(input, 0, 1)
        assert input.size() == (C, N, H, W)

        T = (W - a) // overlap + 1  # tiles_per_channel
        P = N * T * T
        U = torch.zeros(K, C, a, a, device=input.device)
        V = torch.zeros(C, P, a, a, device=input.device)

        for k in range(K):
            for c in range(C):
                U[k, c] = torch.matmul(self.G, torch.matmul(filter[k, c], self.G_T))

        for n in range(N):
            for tH in range(T):
                for tW in range(T):
                    for c in range(C):
                        b = n * (T * T) + tH * T + tW
                        vH = tH * (r - 1)
                        vW = tW * (r - 1)
                        V[c, b] = torch.matmul(
                            self.B_T,
                            torch.matmul(input[c, n, vH : vH + a, vW : vW + a], self.B),
                        )

        M = torch.zeros(K, P, a, a, device=input.device)
        for k in range(K):
            for b in range(P):
                for c in range(C):
                    M[k, b] += U[k, c] * V[c, b]

        out_size = H - r + 1
        Y = torch.zeros(K, N, out_size, out_size)
        for k in range(K):
            for n in range(N):
                for tH in range(T):
                    for tW in range(T):
                        b = n * (T * T) + tH * T + tW
                        oH = tH * m
                        oW = tW * m
                        Y[k, n, oH : oH + m, oW : oW + m] = torch.matmul(
                            self.A_T, torch.matmul(M[k, b], self.A)
                        )

        Y = torch.transpose(Y, 0, 1)
        return Y
