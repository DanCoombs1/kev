"""Loader throughput, padding with and without bucketing, and loader_batch.png in runs/explore/.

    uv run python scripts/check_loader.py
"""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from kev.data.curate import read
from kev.data.loader import BucketSampler, ScanDataset, collate, from_patches, load_library

BATCH = 32
WORKERS = 8
OUT = Path(__file__).resolve().parents[1] / "runs" / "explore"
FONT = ImageFont.load_default(size=14)


def padding_efficiency(loader: torch.utils.data.DataLoader, batches: int) -> float:
    real = total = 0
    for n, batch in enumerate(loader):
        real += int(batch["valid"].sum())
        total += batch["valid"].numel()
        if n + 1 == batches:
            break
    return real / total


def batch_sheet(batch: dict) -> None:
    tiles = []
    for n, sample in enumerate(batch["samples"]):
        start = 0
        for view, (rows, cols) in enumerate(batch["grids"][n]):
            pixels = from_patches(batch["patches"][n, start: start + rows * cols], rows, cols)
            start += rows * cols
            image = Image.fromarray(pixels.permute(1, 2, 0).numpy())
            draw = ImageDraw.Draw(image)
            for item in sample["items"]:
                if item["view"] == view and item["box"]:
                    x, y, w, h = item["box"]
                    draw.rectangle([x, y, x + w, y + h], outline=(230, 0, 180), width=3)
                    draw.text((x + 3, y + 2), item["name"], fill=(230, 0, 180), font=FONT)
            note = " (inserted)" if sample["inserted"] else " (clean)" if sample["pair"] else ""
            draw.text((4, 4), f"{sample['dataset']} view {view}{note}", fill=(0, 0, 0), font=FONT)
            image.thumbnail((360, 300))
            tiles.append(image)
    columns = 4
    sheet = Image.new("RGB", (columns * 370, -(-len(tiles) // columns) * 310), "white")
    for n, im in enumerate(tiles):
        sheet.paste(im, ((n % columns) * 370 + 5, (n // columns) * 310 + 5))
    sheet.save(OUT / "loader_batch.png")


def main() -> None:
    lines = read()
    dataset = ScanDataset(lines, "train", train=True, library=load_library("train"))
    sizes = [dataset.size(k) for k in range(len(dataset))]
    print(f"train: {len(dataset.lines):,} scans -> {len(dataset):,} samples a pass "
          f"({sum(1 for _, insert in dataset.entries if insert):,} with an inserted threat)")

    loader = torch.utils.data.DataLoader(dataset, batch_sampler=BucketSampler(sizes, BATCH, zoom=True), collate_fn=collate,
                                         num_workers=WORKERS, persistent_workers=True)
    batches = iter(loader)
    next(batches)  # workers start up
    start, n = time.perf_counter(), 0
    for batch in batches:
        n += len(batch["samples"])
        if n >= 40 * BATCH:
            break
    print(f"{n / (time.perf_counter() - start):.0f} samples/s with {WORKERS} workers")

    random_batches = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=WORKERS)
    print(f"real patches per padded patch: {padding_efficiency(loader, 20):.0%} bucketed, "
          f"{padding_efficiency(random_batches, 20):.0%} random")

    samples = [dataset[k] for k, (i, insert) in enumerate(dataset.entries) if insert][:3]
    samples += [dataset[int(k)] for k in np.random.default_rng(0).choice(len(dataset), 9, replace=False)]
    batch_sheet(collate(samples))


if __name__ == "__main__":
    main()
