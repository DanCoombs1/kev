"""Scan formats, bag contents, item scale per scanner, and a clean-vs-threat shortcut check.

    uv run python scripts/explore_datasets.py
"""

import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from kev.data.sources import all_scans, image_size, scan_format
from kev.metrics import auc

PATCH = 16
SHARED = ["gun", "knife", "scissors", "pliers", "wrench", "hammer", "screwdriver", "lighter", "battery", "powerbank", "handcuffs", "bullet"]


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


def fullness(path: Path) -> float:
    with Image.open(path) as im:
        grey = np.asarray(im.convert("L").resize((200, 150)))
    return float((grey < 220).mean())


def main() -> None:
    scans = list(all_scans())
    for s in scans:
        if s.sizes is None:
            s.sizes = [image_size(v) for v in s.views]

    print("== formats")
    groups = defaultdict(list)
    for s in scans:
        groups[(s.dataset, scan_format(s.sizes[0][1]))].append(s)
    print(f"{'dataset':<11} {'format':<7} {'scans':>7} {'share':>6}  {'median size':>12}  {'height p5-p95':>14}  {'width p5-p95':>13}")
    for (dataset, f), group in sorted(groups.items()):
        widths = [w for s in group for w, _ in s.sizes]
        heights = [h for s in group for _, h in s.sizes]
        share = len(group) / sum(1 for s in scans if s.dataset == dataset)
        print(f"{dataset:<11} {f:<7} {len(group):>7,} {share:>6.0%}  {statistics.median(widths):>5.0f} x {statistics.median(heights):<4.0f}"
              f"  {percentile(heights, 5):>6} - {percentile(heights, 95):<5}  {percentile(widths, 5):>5} - {percentile(widths, 95):<5}")

    print("\n== bag contents")
    print(f"{'dataset':<11} {'split':<12} {'scans':>7} {'clean':>7}  items per bag")
    for (dataset, split), _ in sorted(Counter((s.dataset, s.split) for s in scans).items()):
        members = [s for s in scans if s.dataset == dataset and s.split == split]
        per_bag = Counter(len([i for i in s.items if i.view == 0]) for s in members)
        spread = ", ".join(f"{k}: {v:,}" for k, v in sorted(per_bag.items()) if k > 0)
        print(f"{dataset:<11} {split:<12} {len(members):>7,} {per_bag.get(0, 0):>7,}  {spread}")

    print("\n== median item size (sqrt of box area, px)")
    sizes = defaultdict(list)
    short_sides = defaultdict(list)
    for s in scans:
        column = f"{s.dataset}/{scan_format(s.sizes[0][1])}"
        for item in s.items:
            if item.box is None:
                continue
            w, h = item.box[2], item.box[3]
            short_sides[column].append(min(w, h))
            if item.label.lower() in SHARED:
                sizes[(item.label.lower(), column)].append((w * h) ** 0.5)
    columns = sorted({c for _, c in sizes})
    print(f"{'item':<12}" + "".join(f"{c:>16}" for c in columns))
    for item in SHARED:
        row = [statistics.median(sizes[(item, c)]) if len(sizes[(item, c)]) >= 20 else None for c in columns]
        print(f"{item:<12}" + "".join(f"{v:>16.0f}" if v else f"{'-':>16}" for v in row))
    reference = "pidray/medium"
    print(f"{'vs ' + reference:<12}", end="")
    for c in columns:
        ratios = [statistics.median(sizes[(i, c)]) / statistics.median(sizes[(i, reference)])
                  for i in SHARED if len(sizes[(i, c)]) >= 20 and len(sizes[(i, reference)]) >= 20]
        print(f"{statistics.median(ratios):>15.2f}x" if ratios else f"{'-':>16}", end="")
    print()
    print(f"{'< 1 patch':<12}" + "".join(f"{sum(v < PATCH for v in short_sides[c]) / len(short_sides[c]):>16.1%}" for c in columns))

    print("\n== clean vs threat from fullness and image area alone (AUC, 0.5 = no signal)")
    print(f"{'dataset':<11} {'bags':>12}  {'fullness clean / threat':>24} {'AUC':>6}  {'image area AUC':>15}")
    rng = random.Random(0)
    for dataset in ["dvxray", "stcray", "iedxray"]:
        members = [s for s in scans if s.dataset == dataset]
        clean = rng.sample([s for s in members if not s.items], min(600, sum(1 for s in members if not s.items)))
        threat = rng.sample([s for s in members if s.items], 600)
        full_clean = [fullness(s.views[0]) for s in clean]
        full_threat = [fullness(s.views[0]) for s in threat]
        area = lambda s: s.sizes[0][0] * s.sizes[0][1]
        print(f"{dataset:<11} {len(clean):>5} + {len(threat):<5}  {statistics.median(full_clean):>11.0%} / {statistics.median(full_threat):<10.0%}"
              f" {auc(full_threat, full_clean):>6.2f}  {auc([area(s) for s in threat], [area(s) for s in clean]):>15.2f}")


if __name__ == "__main__":
    main()
