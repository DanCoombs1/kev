"""Pretrains the image encoder as a masked autoencoder: hide 75% of each scan's patches, show the encoder the rest,
and have a small decoder redraw the hidden ones. The decoder is thrown away afterwards.

    uv run python -m kev.pretrain.image              (trains, resuming if runs/image/checkpoint.pt exists)
    uv run python -m kev.pretrain.image --steps 200  (a short timed trial that saves nothing)
    uv run python -m kev.pretrain.image --layers 12 --stop-after 5 --run image_12   (a size comparison)
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from kev.data.curate import read
from kev.data.loader import ROUND_TO, BucketSampler, ScanDataset, collate, from_patches
from kev.data.prepare import PATCH
from kev.data.sources import DATA
from kev.device import pick_device
from kev.model.blocks import WIDTH, Block
from kev.model.image import LAYERS, MAX_GRID, VIEWS, ImageEncoder, absorption

SHOWN = 0.25
DECODER_WIDTH = 256
DECODER_HEADS = 4
DECODER_LAYERS = 2
SPREAD_FLOOR = 0.01  # added to each patch's variance, so nearly blank patches aren't blown up into noise
POSITION_KEYS = ("patches", "row", "col", "view")

RUN = DATA.parent / "runs" / "image"
BATCH = 16
PASSES = 50
PEAK_LR = 1.5e-4
WARMUP = 0.05
WEIGHT_DECAY = 0.05
WORKERS = 8
VALIDATION_SCANS = 512
PICTURE_EVERY = 10
MEMORY_FRACTION = 0.55  # of what macOS recommends for the GPU (~20 GB): past it the run stops instead of swapping
PATIENCE = 5  # passes without a real gain on validation before stopping early
MIN_PASSES = 15  # never before this: early passes are still warming up, and a plateau there means stuck, not done
MIN_GAIN = 0.002
CACHE_LIMIT = 14e9  # the GPU's cache keeps blocks for every batch size it has seen; let it grow to this, then empty it


def choose_shown(valid: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """A random quarter of each scan's real patches: their places in the row, padded to a round length, and
    which of those places are real."""
    noise = torch.rand(valid.shape, generator=generator)
    noise[~valid] = 2.0  # padding sorts last, so it's never picked
    order = noise.argsort(dim=1)
    counts = (valid.sum(dim=1) * SHOWN).ceil().clamp(min=1).long()
    length = min(-(-int(counts.max()) // ROUND_TO) * ROUND_TO, valid.shape[1])
    return order[:, :length], torch.arange(length) < counts[:, None]


def take(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.gather(1, index)
    return x.gather(1, index[..., None].expand(-1, -1, x.shape[-1]))


def target(patches: torch.Tensor) -> torch.Tensor:
    """Each patch's pattern, not its brightness: absorption less its average, over its spread."""
    x = absorption(patches)
    return (x - x.mean(dim=-1, keepdim=True)) / (x.var(dim=-1, keepdim=True) + SPREAD_FLOOR).sqrt()


def redraw(predicted: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
    """Predicted patterns back into uint8 pixels, using each true patch's average and spread (for pictures)."""
    x = absorption(patches)
    spread = (x.var(dim=-1, keepdim=True) + SPREAD_FLOOR).sqrt()
    return ((1 - (predicted * spread + x.mean(dim=-1, keepdim=True))) * 255).clamp(0, 255).round().to(torch.uint8)


class MaskedAutoencoder(nn.Module):
    def __init__(self, encoder: ImageEncoder | None = None):
        super().__init__()
        self.encoder = encoder or ImageEncoder()
        self.bridge = nn.Linear(WIDTH, DECODER_WIDTH)
        self.hidden = nn.Parameter(torch.zeros(DECODER_WIDTH))  # stands in for every hidden patch
        self.rows = nn.Parameter(torch.zeros(MAX_GRID, DECODER_WIDTH))
        self.cols = nn.Parameter(torch.zeros(MAX_GRID, DECODER_WIDTH))
        self.views = nn.Parameter(torch.zeros(VIEWS, DECODER_WIDTH))
        for table in (self.hidden, self.rows, self.cols, self.views):
            nn.init.normal_(table, std=0.02)
        self.blocks = nn.ModuleList(Block(DECODER_WIDTH, DECODER_HEADS) for _ in range(DECODER_LAYERS))
        self.norm = nn.LayerNorm(DECODER_WIDTH)
        self.pixels = nn.Linear(DECODER_WIDTH, 3 * PATCH * PATCH)

    def forward(self, batch: dict, index: torch.Tensor, shown: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicted patterns for every patch in the row, and which patches the encoder was shown."""
        seen = [take(batch[k], index) for k in POSITION_KEYS]
        encoded = self.bridge(self.encoder(*seen, shown))
        was_shown = torch.zeros_like(batch["valid"]).scatter(1, index, shown)
        x = torch.zeros(*batch["valid"].shape, DECODER_WIDTH, device=encoded.device, dtype=encoded.dtype)
        x = x.scatter(1, index[..., None].expand(-1, -1, DECODER_WIDTH), encoded)
        x = torch.where(was_shown[..., None], x, self.hidden)
        x = x + self.rows[batch["row"]] + self.cols[batch["col"]] + self.views[batch["view"]]
        for block in self.blocks:
            x = block(x, batch["valid"])
        return self.pixels(self.norm(x)), was_shown


def masked_loss(predicted: torch.Tensor, patches: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    return ((predicted - target(patches)) ** 2).mean(dim=-1)[hidden].mean()


def learning_rate(step: int, total: int) -> float:
    """A linear warm-up, then a cosine curve down to a tenth of the peak."""
    warm = max(1, int(WARMUP * total))
    if step < warm:
        return PEAK_LR * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return PEAK_LR * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def passes_since_gain(val_losses: list[float]) -> int:
    """Passes since the validation loss last beat its best by at least MIN_GAIN; the first entry is before training."""
    best, best_at = val_losses[0], 0
    for n, loss in enumerate(val_losses[1:], 1):
        if loss < best - MIN_GAIN:
            best, best_at = loss, n
    return len(val_losses) - 1 - best_at


def should_stop(val_losses: list[float]) -> bool:
    return len(val_losses) - 1 >= MIN_PASSES and passes_since_gain(val_losses) >= PATIENCE


def step_loss(model: MaskedAutoencoder, batch: dict, device: torch.device, generator: torch.Generator | None = None):
    index, shown = choose_shown(batch["valid"], generator)
    on_device = {k: batch[k].to(device) for k in (*POSITION_KEYS, "valid")}
    predicted, was_shown = model(on_device, index.to(device), shown.to(device))
    return masked_loss(predicted, on_device["patches"], on_device["valid"] & ~was_shown), predicted, was_shown


@torch.no_grad()
def validate(model: MaskedAutoencoder, batches: list[dict], device: torch.device) -> float:
    model.eval()
    losses = [step_loss(model, b, device, torch.Generator().manual_seed(n))[0].item() for n, b in enumerate(batches)]
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def picture(model: MaskedAutoencoder, batch: dict, device: torch.device, path) -> None:
    """Each validation scan, the quarter shown, and the hidden rest redrawn."""
    _, predicted, was_shown = step_loss(model, batch, device, torch.Generator().manual_seed(0))
    patches = batch["patches"].to(device)
    rows = [patches, torch.where(was_shown[..., None], patches, torch.full_like(patches, 200)),
            torch.where(was_shown[..., None], patches, redraw(predicted, patches))]
    tiles = []
    for n in range(min(6, len(batch["grids"]))):
        r, c = batch["grids"][n][0]
        tiles.append([Image.fromarray(from_patches(row[n, : r * c].cpu(), r, c).permute(1, 2, 0).numpy()) for row in rows])
    for row in tiles:
        for im in row:
            im.thumbnail((300, 300))
    height = max(im.height for row in tiles for im in row) + 10
    sheet = Image.new("RGB", (3 * 310, len(tiles) * height), "white")
    for r, row in enumerate(tiles):
        for c, im in enumerate(row):
            sheet.paste(im, (c * 310 + 5, r * height + 5))
    sheet.save(path)


def train(trial_steps: int | None = None, layers: int = LAYERS, stop_after: int | None = None, run: Path = RUN) -> None:
    device = pick_device()
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(MEMORY_FRACTION)
    torch.manual_seed(0)
    lines = read()
    data = ScanDataset(lines, "train", train=True)
    sampler = BucketSampler([data.size(k) for k in range(len(data))], BATCH, zoom=True)
    loader = torch.utils.data.DataLoader(data, batch_sampler=sampler, collate_fn=collate, num_workers=WORKERS,
                                         persistent_workers=True)
    val = ScanDataset(lines, "val", train=False)
    picks = sorted(random.Random(0).sample(range(len(val)), VALIDATION_SCANS))
    val_sampler = BucketSampler([val.size(k) for k in picks], BATCH, shuffle=False)
    val_batches = [collate([val[picks[i]] for i in b]) for b in val_sampler]

    model = MaskedAutoencoder(ImageEncoder(layers=layers)).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
                                  lr=PEAK_LR, betas=(0.9, 0.95))
    total = PASSES * len(sampler)
    first_pass, step, val_losses = 0, 0, []
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = run / "checkpoint.pt"
    if trial_steps is None and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        first_pass, step, val_losses = saved["pass"] + 1, saved["step"], saved["val_losses"]
        print(f"resuming after pass {first_pass} (step {step:,})")
    print(f"{len(data):,} training scans, {total:,} steps over {PASSES} passes, {len(val_batches)} validation batches",
          flush=True)

    with open(run / "log.jsonl", "a") as log:
        def record(entry: dict) -> None:
            if trial_steps is None:
                log.write(json.dumps(entry) + "\n")
                log.flush()
            print("  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in entry.items()), flush=True)

        if first_pass == 0 and trial_steps is None:
            val_losses.append(validate(model, val_batches, device))
            record({"pass": 0, "val_loss": val_losses[-1]})
        start, patches, running = time.perf_counter(), 0, []
        for n in range(first_pass, PASSES):
            sampler.epoch = n  # a different shuffle and set of zooms each pass, also after resuming
            for batch in loader:
                lr = learning_rate(step, total)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                loss = step_loss(model, batch, device)[0]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if device.type == "mps" and torch.mps.driver_allocated_memory() > CACHE_LIMIT:
                    torch.mps.empty_cache()
                step += 1
                patches += int(batch["valid"].sum())
                running.append(loss.detach())
                if step % 200 == 0 or step == trial_steps:
                    elapsed = time.perf_counter() - start
                    record({"step": step, "loss": torch.stack(running).mean().item(), "lr": lr,
                            "patches_per_s": round(patches / elapsed), "minutes": round(elapsed / 60, 1),
                            "gpu_gb": round(torch.mps.driver_allocated_memory() / 1e9, 1)})
                    running = []
                if step == trial_steps:
                    print(f"about {len(sampler) * elapsed / step / 60:.0f} minutes a pass")
                    return
            val_losses.append(validate(model, val_batches, device))
            record({"pass": n + 1, "val_loss": val_losses[-1]})
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "pass": n, "step": step,
                        "val_losses": val_losses}, checkpoint)
            torch.save({"model": model.encoder.state_dict(), "pass": n + 1}, run / "encoder.pt")
            since = passes_since_gain(val_losses)
            if val_losses[-1] == min(val_losses):  # the lowest loss, however small the gain; MIN_GAIN is only for stopping
                torch.save({"model": model.encoder.state_dict(), "pass": n + 1}, run / "encoder_best.pt")
            if (n + 1) % PICTURE_EVERY == 0 or n + 1 == PASSES or should_stop(val_losses):
                picture(model, val_batches[0], device, run / f"redrawn_pass_{n + 1}.png")
            if should_stop(val_losses):
                record({"stopped_early_after_pass": n + 1, "best_pass": n + 1 - since})
                return
            if n + 1 == stop_after:
                return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, help="a short timed trial that saves nothing")
    parser.add_argument("--layers", type=int, default=LAYERS, help="encoder blocks")
    parser.add_argument("--stop-after", type=int, help="passes to run; the learning-rate schedule is still planned over PASSES")
    parser.add_argument("--run", default=RUN.name, help="folder under runs/")
    args = parser.parse_args()
    train(args.steps, args.layers, args.stop_after, RUN.parent / args.run)


if __name__ == "__main__":
    main()
