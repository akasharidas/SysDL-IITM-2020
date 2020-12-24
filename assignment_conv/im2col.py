import torch
from torch._C import device


class Im2Col_Conv2d(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _input, _filter):
        assert _input.shape[1] == _filter.shape[1]
        N, C, H, W = tuple(_input.shape)
        M, _, R, S = tuple(_filter.shape)

        sx = H - R + 1
        sy = W - S + 1

        # expand input
        col = torch.zeros((N, C, R, S, sx, sy), device=_input.device)
        for y in range(R):
            y_max = y + sx
            for x in range(S):
                x_max = x + sy
                col[:, :, y, x, :, :] = _input[:, :, y:y_max, x:x_max]
        col = col.permute(0, 4, 5, 1, 2, 3).reshape(N * sx * sy, -1).T

        # expand filter
        fil = torch.zeros((M, C * R * S), device=_input.device)
        for m in range(M):
            fil[m, :] = _filter[m, :].reshape(1, -1)

        # multiply and convert back to image
        return torch.cat(torch.split(torch.matmul(fil, col), sx * sy, dim=1)).reshape(
            N, M, sx, sy
        )
