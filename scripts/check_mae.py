"""Can the masked autoencoder memorise a handful of scans? If it can't, something is broken. Writes mae_memorise.png
in runs/explore/: each scan, the quarter the encoder was shown, and the hidden rest redrawn.

    uv run python scripts/check_mae.py [steps]
"""

import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from kev.data.curate import read
from kev.data.loader import ScanDataset, collate, from_patches
from kev.device import pick_device
from kev.pretrain.image import MaskedAutoencoder, choose_shown, masked_loss, redraw

PER_DATASET = 2
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OUT = Path(__file__).resolve().parents[1] / "runs" / "explore"
GREY = 200


def first_view(batch: dict, row: torch.Tensor, n: int) -> Image.Image:
    rows, cols = batch["grids"][n][0]
    pixels = from_patches(row[: rows * cols].cpu(), rows, cols)
    return Image.fromarray(pixels.permute(1, 2, 0).numpy())


def main() -> None:
    device = pick_device()
    torch.manual_seed(0)
    rng = random.Random(0)
    dataset = ScanDataset(read(), "train", train=False)
    picks = []
    for name in ("stcray", "pidray", "dvxray", "iedxray"):
        picks += rng.sample([k for k, (i, _) in enumerate(dataset.entries) if dataset.lines[i]["dataset"] == name], PER_DATASET)
    batch = collate([dataset[k] for k in picks])
    on_device = {k: batch[k].to(device) for k in ("patches", "row", "col", "view", "valid")}

    model = MaskedAutoencoder().to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters "
          f"({sum(p.numel() for p in model.encoder.parameters()) / 1e6:.1f}M encoder, the rest is the throwaway decoder)")
    index, shown = choose_shown(batch["valid"], torch.Generator().manual_seed(1))
    _, was_shown = model(on_device, index.to(device), shown.to(device))
    flat = masked_loss(torch.zeros_like(on_device["patches"], dtype=torch.float), on_device["patches"], on_device["valid"] & ~was_shown)
    print(f"guessing a flat patch everywhere scores {flat.item():.3f}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    start = time.perf_counter()
    for step in range(STEPS + 1):
        index, shown = choose_shown(batch["valid"])
        predicted, was_shown = model(on_device, index.to(device), shown.to(device))
        loss = masked_loss(predicted, on_device["patches"], on_device["valid"] & ~was_shown)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % max(50, STEPS // 10) == 0:
            print(f"step {step:3d}  loss {loss.item():.3f}")
    print(f"({time.perf_counter() - start:.0f}s)")

    index, shown = choose_shown(batch["valid"], torch.Generator().manual_seed(1))
    with torch.no_grad():
        predicted, was_shown = model(on_device, index.to(device), shown.to(device))
    patches = on_device["patches"]
    visible = torch.where(was_shown[..., None], patches, torch.full_like(patches, GREY))
    redrawn = torch.where(was_shown[..., None], patches, redraw(predicted, patches))
    tiles = [[first_view(batch, row[n], n) for row in (patches, visible, redrawn)] for n in range(len(picks))]
    for row in tiles:
        for im in row:
            im.thumbnail((300, 300))
    height = max(im.height for row in tiles for im in row) + 10
    sheet = Image.new("RGB", (3 * 310, len(tiles) * height), "white")
    for r, row in enumerate(tiles):
        for c, im in enumerate(row):
            sheet.paste(im, (c * 310 + 5, r * height + 5))
    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT / "mae_memorise.png")
    print(f"columns: scan, the quarter shown, hidden patches redrawn -> {OUT / 'mae_memorise.png'}")


if __name__ == "__main__":
    main()
