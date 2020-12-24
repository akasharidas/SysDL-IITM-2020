import torch
import time
from statistics import mean
from collections import defaultdict
from tqdm import tqdm
import json

from direct import Direct_Conv2d
from im2col import Im2Col_Conv2d
from winograd import Winograd_Conv2d
from fft import FFT_Conv2d


def run_and_time(name, conv2d, _input, _filter):
    times = []
    iters = 1 if name == "direct" else 1

    for _ in tqdm(range(iters), desc=f"Running benchmark for {name}"):
        start_time = time.time()
        out = conv2d(_input, _filter)
        times.append(time.time() - start_time)
        del out

    return mean(times)


def test_implementation(conv2d, _input, _filter):
    reference = torch.nn.functional.conv2d(_input, _filter)
    return torch.allclose(reference, conv2d(_input, _filter))


if __name__ == "__main__":
    convs = {
        "direct": Direct_Conv2d(),
        "im2col": Im2Col_Conv2d(),
        "winograd": Winograd_Conv2d(),
        "fft": FFT_Conv2d(),
    }

    devices = ["cpu", "cuda"]
    results = defaultdict(dict)

    N, C, H, W, M, R, S = 8, 3, 32, 32, 2, 3, 3
    test_i = torch.rand(N, C, H, W, dtype=torch.float)
    test_f = torch.rand(M, C, R, S, dtype=torch.float)

    with torch.no_grad():

        # check that the outputs match the reference PyTorch implementation
        print("Testing correctness of implementations...")
        for _, conv2d in convs.items():
            assert test_implementation(conv2d, test_i, test_f)
        print("Tests passed.")

        for device in devices:
            print(f"Running benchmarks on {device}...")
            test_i, test_f = test_i.to(device), test_f.to(device)
            for name, conv2d in convs.items():
                conv2d.to(device)
                results[device][name] = run_and_time(name, conv2d, test_i, test_f)

    with open("results.pkl", "w") as f:
        json.dump(dict(results), f, indent=4)
