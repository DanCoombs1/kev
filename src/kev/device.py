"""Pick where kev's maths runs: an NVIDIA GPU, the Mac's GPU, or the CPU."""

import torch


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():  # Apple silicon GPU: what kev trains on
        return torch.device("mps")
    if torch.cuda.is_available():  # NVIDIA GPU, in case the code is ever run on one
        return torch.device("cuda")
    return torch.device("cpu")
