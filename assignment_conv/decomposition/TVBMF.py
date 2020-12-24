import torch
import numpy as np
from scipy.optimize import minimize_scalar


def EVBMF(Y, sigma2=None, H=None):
    L, M = Y.shape  # has to be L<=M

    if H is None:
        H = L

    alpha = L / M
    tauubar = 2.5129 * np.sqrt(alpha)

    # SVD of the input matrix, max rank of H
    U, s, V = torch.svd(Y)
    U = U[:, :H]
    s = s[:H]
    V[:H].t_()

    # Calculate residual
    residual = 0.0
    if H < L:
        residual = torch.sum(torch.sum(Y ** 2) - torch.sum(s ** 2))

    # Estimation of the variance when sigma2 is unspecified
    if sigma2 is None:
        xubar = (1 + tauubar) * (1 + alpha / tauubar)
        eH_ub = int(np.min([np.ceil(L / (1 + alpha)) - 1, H])) - 1
        upper_bound = (torch.sum(s ** 2) + residual) / (L * M)
        lower_bound = np.max(
            [s[eH_ub + 1] ** 2 / (M * xubar), torch.mean(s[eH_ub + 1 :] ** 2) / M]
        )

        scale = 1.0  # /lower_bound
        s = s * np.sqrt(scale)
        residual = residual * scale
        lower_bound = lower_bound * scale
        upper_bound = upper_bound * scale

        sigma2_opt = minimize_scalar(
            EVBsigma2,
            args=(L, M, s, residual, xubar),
            bounds=[lower_bound.item(), upper_bound.item()],
            method="Bounded",
        )
        sigma2 = sigma2_opt.x

    # Threshold gamma term
    threshold = np.sqrt(M * sigma2 * (1 + tauubar) * (1 + alpha / tauubar))

    pos = torch.sum(s > threshold)
    if pos == 0:
        return np.array([])

    # Formula (15) from [2]
    d = torch.mul(
        s[:pos] / 2,
        1
        - (L + M) * sigma2 / s[:pos] ** 2
        + torch.sqrt(
            (1 - ((L + M) * sigma2) / s[:pos] ** 2) ** 2
            - (4 * L * M * sigma2 ** 2) / s[:pos] ** 4
        ),
    )

    return torch.diag(d)


def EVBsigma2(sigma2, L, M, s, residual, xubar):
    H = len(s)

    alpha = L / M
    x = s ** 2 / (M * sigma2)

    z1 = x[x > xubar]
    z2 = x[x <= xubar]
    tau_z1 = tau(z1, alpha)

    term1 = torch.sum(z2 - torch.log(z2))
    term2 = torch.sum(z1 - tau_z1)
    term3 = torch.sum(torch.log((tau_z1 + 1) / z1))
    term4 = alpha * torch.sum(torch.log(tau_z1 / alpha + 1))

    obj = (
        term1
        + term2
        + term3
        + term4
        + residual / (M * sigma2)
        + (L - H) * np.log(sigma2)
    )

    return obj


def tau(x, alpha):
    return 0.5 * (x - (1 + alpha) + torch.sqrt((x - (1 + alpha)) ** 2 - 4 * alpha))

