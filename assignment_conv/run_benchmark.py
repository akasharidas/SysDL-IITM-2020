import torch
import time
from statistics import mean
from collections import defaultdict
from tqdm import tqdm
import json
import wandb

from direct import Direct_Conv2d
from im2col import Im2Col_Conv2d
from winograd import Winograd_Conv2d
from fft import FFT_Conv2d
from decomp import Tucker_Conv2d
from decomp import CP_Conv2d


def run_and_time(name, model, _input, _filter):
    times = []
    iters = 1 if name == "direct" else 5

    for _ in tqdm(range(iters), desc=f"Running benchmark for {name}"):
        start_time = time.time()
        out = model(_input, _filter)
        times.append(time.time() - start_time)
        del out

    return mean(times)


def test_implementation(name, model, _input, _filter):
    reference = torch.nn.functional.conv2d(_input, _filter)

    """the tucker and cp models discard the original weights and initialize 
    new weights for each decomposed block, therefore we only check the shape"""
    if name in ["tucker", "cp"]:
        return reference.shape == model(_input, _filter).shape

    return torch.allclose(reference, model(_input, _filter))


def run(config, convs, devices, test=False):
    wandb.init(project="sysdl-assignment7")

    results = defaultdict(dict)
    N, C, H, W, M, R, S = (
        config["N"],
        config["C"],
        config["H"],
        config["W"],
        config["M"],
        config["R"],
        config["S"],
    )
    test_i = torch.rand(N, C, H, W, dtype=torch.float)
    test_f = torch.rand(M, C, R, S, dtype=torch.float)

    with torch.no_grad():

        if test:
            # check that the outputs match the reference PyTorch implementation
            print("Testing correctness of implementations...")
            for name, conv2d in convs.items():
                model = (
                    conv2d(C, M, R, S, test_f) if name in ["tucker", "cp"] else conv2d()
                )
                assert test_implementation(name, model, test_i, test_f)
            print("Tests passed.")

        for device in devices:
            print(f"Running benchmarks on {device}...")
            test_i, test_f = test_i.to(device), test_f.to(device)
            for name, conv2d in convs.items():
                model = (
                    conv2d(C, M, R, S, test_f) if name in ["tucker", "cp"] else conv2d()
                )
                model.to(device)
                results[name][device] = run_and_time(name, model, test_i, test_f)

    wandb.log({"results": dict(results), "config": config})


if __name__ == "__main__":
    wandb.login(key="df416cf0e6b9361efc64aa08d4715af979c8d070")

    convs = {
        "im2col": Im2Col_Conv2d,
        "winograd": Winograd_Conv2d,
        "fft": FFT_Conv2d,
        "tucker": Tucker_Conv2d,
        "cp": CP_Conv2d,
        "direct": Direct_Conv2d,
    }
    devices = ["cpu", "cuda"]

    config = dict(N=8, C=3, H=32, W=32, M=2, R=3, S=3)
    run(config, convs, devices)
