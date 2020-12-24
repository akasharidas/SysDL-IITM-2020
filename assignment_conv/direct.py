import torch


class Direct_Conv2d(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, _input, _filter):
        assert _input.shape[1] == _filter.shape[1]
        N, C, H, W = tuple(_input.shape)
        M, _, R, S = tuple(_filter.shape)
        _output = torch.zeros(N, M, H - R + 1, W - S + 1, device=_input.device)

        for n in range(N):
            for c in range(C):
                for h in range(H - R + 1):
                    for w in range(W - S + 1):
                        for m in range(M):
                            for r in range(R):
                                for s in range(S):
                                    _output[n][m][h][w] += (
                                        _input[n][c][h + r][w + s] * _filter[m][c][r][s]
                                    )

        return _output
