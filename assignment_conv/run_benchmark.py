import torch
import time
from statistics import mean
from collections import defaultdict
from tqdm import tqdm
import wandb

from direct import Direct_Conv2d
from im2col import Im2Col_Conv2d
from winograd import Winograd_Conv2d
from fft import FFT_Conv2d
from decomp import Tucker_Conv2d
from decomp import CP_Conv2d
from config import configs


def run_and_time(name, model, _input, _filter):
    times = []
    iters = 1 if name == "direct" else 3 if name == "winograd" else 5

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
    print(f"\nCurrent config: {config}")

    results = defaultdict(dict)
    C, H, W, M, R, S = (
        config["C"],
        config["H"],
        config["W"],
        config["M"],
        config["R"],
        config["S"],
    )

    with torch.no_grad():
        if test:
            # check that the outputs match the reference PyTorch implementation
            print("Testing correctness of implementations...")
            img = torch.rand(8, 3, 16, 16, dtype=torch.float)
            fil = torch.rand(2, 3, 3, 3, dtype=torch.float)
            for name, conv2d in convs.items():
                model = (
                    conv2d(C, M, R, S, fil) if name in ["tucker", "cp"] else conv2d()
                )
                assert test_implementation(name, model, img, fil)
            print("Tests passed.")

        for device in devices:
            print(f"\nRunning benchmarks on {device}...")
            N = 8 if device == "cpu" else 128
            img = torch.rand(N, C, H, W, dtype=torch.float, device=device)
            fil = torch.rand(M, C, R, S, dtype=torch.float, device=device)
            for name, conv2d in convs.items():
                model = (
                    conv2d(C, M, R, S, fil) if name in ["tucker", "cp"] else conv2d()
                )
                model.to(device)
                results[name][device] = run_and_time(name, model, img, fil)

    wandb.log({"results": dict(results), "config": config})


if __name__ == "__main__":
    wandb.login(key="df416cf0e6b9361efc64aa08d4715af979c8d070")

    convolutions = {
        "im2col": Im2Col_Conv2d,
        "winograd": Winograd_Conv2d,
        "fft": FFT_Conv2d,
        "tucker": Tucker_Conv2d,
        "cp": CP_Conv2d,
        "direct": Direct_Conv2d,
    }
    devices = ["cpu", "cuda"]

    for c, h, w, m, r, s in configs:
        config = dict(C=c, H=h, W=w, M=m, R=r, S=s)

        invalid = []
        if not (config["R"] == 3 and config["S"] == 3):
            invalid.append("winograd")
        if not ((config["H"] == config["W"]) and (config["H"] % 2 == 0)):
            invalid.append("winograd")
        if config["H"] > 32 or config["W"] > 32 or config["C"] > 3:
            invalid.append("direct")

        convs = {k: v for k, v in convolutions.items() if k not in invalid}
        run(config, convs, devices)
