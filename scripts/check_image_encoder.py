"""Speed and memory of the image encoder on real training batches, with every patch and with a quarter of them.

    uv run python scripts/check_image_encoder.py
"""

import time

import torch

from kev.data.curate import read
from kev.data.loader import ROUND_TO, BucketSampler, ScanDataset, collate, load_library
from kev.device import pick_device
from kev.model.image import ImageEncoder

BATCH = 16
STEPS = 10


def keep_quarter(batch: dict, generator: torch.Generator) -> dict:
    """A random quarter of each scan's real patches, repacked from the start of the row."""
    keep = []
    for valid in batch["valid"]:
        real = valid.nonzero().squeeze(1)
        keep.append(real[torch.randperm(len(real), generator=generator)[: max(1, len(real) // 4)]])
    length = -(-max(map(len, keep)) // ROUND_TO) * ROUND_TO
    out = {k: torch.zeros((len(keep), length) + batch[k].shape[2:], dtype=batch[k].dtype) for k in ("patches", "row", "col", "view", "valid")}
    for i, idx in enumerate(keep):
        for k in out:
            out[k][i, : len(idx)] = batch[k][i, idx]
    return out


def run(model: ImageEncoder, batches: list[dict], device: torch.device) -> tuple[float, float, float]:
    """Times a second pass over the batches, once the GPU has built kernels for each shape."""
    for timed in (False, True):
        torch.mps.synchronize()
        start = time.perf_counter()
        for b in batches:
            model(*(b[k].to(device) for k in ("patches", "row", "col", "view", "valid"))).sum().backward()
        torch.mps.synchronize()
    seconds = (time.perf_counter() - start) / len(batches)
    patches = sum(int(b["valid"].sum()) for b in batches) / len(batches)
    return seconds, patches / seconds, torch.mps.driver_allocated_memory() / 1e9


def main() -> None:
    device = pick_device()
    lines = read()
    dataset = ScanDataset(lines, "train", train=True, library=load_library("train"))
    sampler = BucketSampler([dataset.size(k) for k in range(len(dataset))], BATCH, zoom=True)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=8)
    batches = []
    for b in loader:
        batches.append(b)
        if len(batches) == STEPS:
            break
    model = ImageEncoder().to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    print(f"batches of {BATCH} scans, longest {max(b['patches'].shape[1] for b in batches)} patches\n")

    seconds, rate, memory = run(model, batches, device)
    per_pass = len(dataset) / BATCH * seconds / 60
    print(f"every patch:     {seconds * 1000:4.0f} ms a batch, {rate / 1000:.0f}k patches/s, "
          f"~{per_pass:.0f} min a pass over training, {memory:.1f} GB")
    generator = torch.Generator().manual_seed(0)
    seconds, rate, memory = run(model, [keep_quarter(b, generator) for b in batches], device)
    per_pass = len(dataset) / BATCH * seconds / 60
    print(f"a quarter shown: {seconds * 1000:4.0f} ms a batch, {rate / 1000:.0f}k patches/s, "
          f"~{per_pass:.0f} min a pass over training (encoder only), {memory:.1f} GB")


if __name__ == "__main__":
    main()
