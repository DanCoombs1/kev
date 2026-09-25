"""Can a small CNN separate threat bags from clean bags without seeing the threats?

Trains three probes per dataset: on untouched scans (control), with every threat blanked out
(clean bags get blanks of the same sizes in the same relative places), and on bag outlines only.
Trained on the train role, scored on val.

    uv run python scripts/blind_probe.py
"""

import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from kev.data.curate import read
from kev.data.duplicates import content_box
from kev.data.sources import DATA
from kev.device import pick_device
from kev.metrics import auc

SIZE = 96
DATASETS = ["dvxray", "stcray"]
EPOCHS = 15
OUT = Path(__file__).resolve().parents[1] / "runs" / "explore"


def fully_boxed(line: dict) -> bool:
    return all(i["box"] for i in line["items"] if i["view"] == 0)


def choose(lines: list[dict], dataset: str, role: str, rng: random.Random, balance: bool) -> tuple[list, list]:
    pool = [l for l in lines if l["dataset"] == dataset and l["role"] == role]
    clean = [l for l in pool if not l["items"]]
    threat = [l for l in pool if l["items"] and fully_boxed(l)]
    if balance:
        n = min(len(clean), len(threat))
        clean, threat = rng.sample(clean, n), rng.sample(threat, n)
    return clean, threat


def find_bag(path: str) -> tuple[int, int, int, int]:
    with Image.open(DATA / path) as im:
        return content_box(np.asarray(im.convert("L")))


def relative(box: list[float], bag: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bag
    return (box[0] - x0) / (x1 - x0), (box[1] - y0) / (y1 - y0), box[2] / (x1 - x0), box[3] / (y1 - y0)


def absolute(box: tuple[float, float, float, float], bag: tuple[int, int, int, int]) -> list[float]:
    x0, y0, x1, y1 = bag
    return [x0 + box[0] * (x1 - x0), y0 + box[1] * (y1 - y0), box[2] * (x1 - x0), box[3] * (y1 - y0)]


def thumbnail(pixels: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(pixels).resize((SIZE, SIZE), Image.Resampling.BILINEAR))


def outline(thumbnails: np.ndarray) -> np.ndarray:
    bag = thumbnails.mean(axis=3) < 230
    return np.repeat(np.where(bag, 0, 255).astype(np.uint8)[..., None], 3, axis=3)


def prepare(job: tuple) -> tuple[np.ndarray, np.ndarray]:
    path, (x0, y0, x1, y1), boxes = job
    with Image.open(DATA / path) as im:
        pixels = np.asarray(im.convert("RGB")).copy()
    untouched = thumbnail(pixels[y0:y1, x0:x1])
    for x, y, w, h in boxes:
        margin = max(4, 0.1 * max(w, h))
        pixels[max(0, int(y - margin)): int(y + h + margin) + 1, max(0, int(x - margin)): int(x + w + margin) + 1] = 255
    return untouched, thumbnail(pixels[y0:y1, x0:x1])


def prepare_all(clean: list[dict], threat: list[dict], rng: random.Random, pool: Pool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = clean + threat
    bags = pool.map(find_bag, [l["views"][0] for l in lines], chunksize=64)
    bag_of = {l["id"]: b for l, b in zip(lines, bags)}
    threat_places = [[relative(i["box"], bag_of[l["id"]]) for i in l["items"] if i["view"] == 0] for l in threat]
    jobs = []
    for l in clean:
        jobs.append((l["views"][0], bag_of[l["id"]], [absolute(b, bag_of[l["id"]]) for b in rng.choice(threat_places)]))
    for l in threat:
        jobs.append((l["views"][0], bag_of[l["id"]], [i["box"] for i in l["items"] if i["view"] == 0]))
    untouched, blanked = zip(*pool.map(prepare, jobs, chunksize=64))
    labels = np.array([0] * len(clean) + [1] * len(threat), dtype=np.float32)
    return np.stack(untouched), np.stack(blanked), labels


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = [3, 16, 32, 64, 128]
        for c_in, c_out in zip(channels, channels[1:]):
            layers += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.score = nn.Linear(channels[-1], 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.score(self.features(images).mean(dim=(2, 3))).squeeze(1)


def to_tensor(thumbnails: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(thumbnails).permute(0, 3, 1, 2).float() / 255 - 0.5


def scores(model: nn.Module, images: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(images[i: i + 256].to(device)).cpu() for i in range(0, len(images), 256)]).numpy()


def train_and_score(train_x, train_y, val_x, val_y, device, label: str) -> float:
    torch.manual_seed(0)
    model = Probe().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    train_x, train_y, val_x = to_tensor(train_x), torch.from_numpy(train_y), to_tensor(val_x)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_x))
        total = 0.0
        for start in range(0, len(order), 128):
            batch = order[start: start + 128]
            images, answers = train_x[batch].to(device), train_y[batch].to(device)
            flip = torch.rand(len(images), 1, 1, 1, device=device) < 0.5
            images = torch.where(flip, images.flip(3), images)
            loss = loss_fn(model(images), answers)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        val_scores = scores(model, val_x, device)
        val_auc = auc(val_scores[val_y == 1].tolist(), val_scores[val_y == 0].tolist())
        if epoch in (1, 5, 10, EPOCHS):
            print(f"   {label:<9} epoch {epoch:>2}: train loss {total / len(order):.3f}   val AUC {val_auc:.2f}")
    return val_auc


def save_examples(untouched: np.ndarray, blanked: np.ndarray, labels: np.ndarray, dataset: str) -> None:
    rows = []
    for kind in (0, 1):
        idx = np.nonzero(labels == kind)[0][:6]
        rows.append(np.concatenate([np.concatenate([untouched[i], blanked[i]], axis=0) for i in idx], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).resize((6 * SIZE * 2, 4 * SIZE * 2)).save(OUT / f"probe_inputs_{dataset}.png")


def main() -> None:
    device = pick_device()
    lines = read()
    rng = random.Random(0)
    results = {}
    with Pool() as pool:
        for dataset in DATASETS:
            train_clean, train_threat = choose(lines, dataset, "train", rng, balance=True)
            val_clean, val_threat = choose(lines, dataset, "val", rng, balance=False)
            print(f"\n{dataset}: train {len(train_clean):,} clean + {len(train_threat):,} threat, "
                  f"val {len(val_clean):,} clean + {len(val_threat):,} threat")
            tr_untouched, tr_blanked, tr_y = prepare_all(train_clean, train_threat, rng, pool)
            va_untouched, va_blanked, va_y = prepare_all(val_clean, val_threat, rng, pool)
            save_examples(va_untouched, va_blanked, va_y, dataset)
            results[dataset] = (
                train_and_score(tr_untouched, tr_y, va_untouched, va_y, device, "control"),
                train_and_score(tr_blanked, tr_y, va_blanked, va_y, device, "blind"),
                train_and_score(outline(tr_untouched), tr_y, outline(va_untouched), va_y, device, "outline"),
            )
    print(f"\n{'dataset':<9} {'control':>8} {'blind':>8} {'outline':>8}")
    for dataset, (control, blind, shape) in results.items():
        print(f"{dataset:<9} {control:>8.2f} {blind:>8.2f} {shape:>8.2f}")


if __name__ == "__main__":
    main()
