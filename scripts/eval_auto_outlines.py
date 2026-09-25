"""Scores box-derived outlines against the hand-drawn STCray and PIDray outlines (pixel IoU).

An item type passes if its median IoU is at least 0.8.

    uv run python scripts/eval_auto_outlines.py
"""

import random
import statistics
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from kev.data.curate import read
from kev.data.names import kev_name
from kev.data.outlines import bounding_box, box_outline, grabcut_outline, polygon_mask, ratio_outline, true_outlines
from kev.data.sources import DATA

PASS = 0.8
PER_ITEM = 120
METHODS = {"box": box_outline, "ratio": ratio_outline, "grabcut": grabcut_outline}
OUT = Path(__file__).resolve().parents[1] / "runs" / "explore"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return (a & b).sum() / max(1, (a | b).sum())


def score(job: tuple) -> dict[str, float]:
    path, points = job
    pixels = np.asarray(Image.open(DATA / path).convert("RGB"))
    truth = polygon_mask(points, (pixels.shape[1], pixels.shape[0]))
    box = bounding_box(points)
    return {name: iou(method(pixels, box), truth) for name, method in METHODS.items()}


def overlay(pixels: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    edge = mask & ~ndimage.binary_erosion(mask, iterations=2)
    out = pixels.copy()
    out[edge] = colour
    return out


def example_sheet(jobs: list[tuple]) -> None:
    """Truth in green, ratio in red, GrabCut in blue."""
    rows = []
    for path, points in jobs:
        pixels = np.asarray(Image.open(DATA / path).convert("RGB"))
        box = bounding_box(points)
        x, y, w, h = (int(v) for v in box)
        m = 15
        crop = lambda im: im[max(0, y - m): y + h + m, max(0, x - m): x + w + m]
        panels = [crop(overlay(pixels, polygon_mask(points, (pixels.shape[1], pixels.shape[0])), (0, 200, 0))),
                  crop(overlay(pixels, ratio_outline(pixels, box), (230, 0, 0))),
                  crop(overlay(pixels, grabcut_outline(pixels, box), (0, 80, 255)))]
        height = 160
        panels = [np.asarray(Image.fromarray(p).resize((max(1, int(p.shape[1] * height / p.shape[0])), height))) for p in panels]
        gap = np.full((height, 12, 3), 255, np.uint8)
        rows.append(np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1))
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 12), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    Image.fromarray(np.concatenate(rows, axis=0)).save(OUT / "auto_outlines.png")


def main() -> None:
    rng = random.Random(0)
    found = defaultdict(list)
    for line in read():
        if line["role"] == "train" and line["dataset"] in ("stcray", "pidray"):
            for label, points in true_outlines(line):
                if len(points) >= 3:
                    found[(line["dataset"], kev_name(line["dataset"], label))].append((line["views"][0], points))
    keys = sorted(found)
    jobs = {k: rng.sample(found[k], min(PER_ITEM, len(found[k]))) for k in keys}
    with Pool() as pool:
        results = {k: pool.map(score, jobs[k]) for k in keys}

    print(f"{'dataset':<8} {'item':<20} {'n':>4} {'box':>6} {'ratio':>6} {'grabcut':>8}   best")
    passed = defaultdict(list)
    for dataset, item in keys:
        medians = {m: statistics.median(r[m] for r in results[(dataset, item)]) for m in METHODS}
        best = max(("ratio", "grabcut"), key=medians.get)
        verdict = "pass" if medians[best] >= PASS else "fail"
        passed[verdict].append(f"{item} ({dataset})")
        print(f"{dataset:<8} {item:<20} {len(results[(dataset, item)]):>4} {medians['box']:>6.2f} {medians['ratio']:>6.2f} "
              f"{medians['grabcut']:>8.2f}   {best} {verdict}")
    print(f"\npass: {len(passed['pass'])}   fail: {len(passed['fail'])}")

    showcase = ["gun", "knife", "3D-printed gun", "explosive", "lighter", "battery", "scissors", "spray can"]
    picks = [rng.choice(jobs[k]) for k in keys if k[1] in showcase and k[0] == ("pidray" if k[1] == "spray can" else "stcray")]
    example_sheet(picks)


if __name__ == "__main__":
    main()
