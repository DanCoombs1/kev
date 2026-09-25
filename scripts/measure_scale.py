"""Per-dataset resize factors to STCray's pixel scale, from item and bag sizes.

The factor is the median over shared item types of STCray's median size divided by the
dataset's. Bag sizes are printed as a cross-check. COMPASS-XP has no boxes, so it isn't measured.

    uv run python scripts/measure_scale.py
"""

import random
import statistics
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from PIL import Image

from kev.data.curate import read
from kev.data.duplicates import content_box
from kev.data.sources import DATA, scan_format

MIN_ITEMS = 20


def bag_box(path: str) -> tuple[int, int, int, int]:
    with Image.open(DATA / path) as im:
        return content_box(np.asarray(im.convert("L")))


def bag_rulers(lines: list[dict]) -> None:
    # STCray and IEDXray share suitcases, so their bags should measure the same. DvXray's views
    # share the belt axis, though the side view's belt line inflates its bag length.
    rng = random.Random(0)
    groups = defaultdict(list)
    for line in lines:
        if line["dataset"] in ("stcray", "iedxray") and line["role"] == "train":
            groups[f"{line['dataset']}/{scan_format(line['sizes'][0][1])}"].append(line)
    with Pool() as pool:
        print("\nmedian bag long side (px)")
        for column in sorted(groups):
            boxes = pool.map(bag_box, [l["views"][0] for l in rng.sample(groups[column], 400)])
            print(f"   {column:<16} {statistics.median(max(x1 - x0, y1 - y0) for x0, y0, x1, y1 in boxes):>6.0f}")
        bags = rng.sample([l for l in lines if l["dataset"] == "dvxray"], 400)
        tops = pool.map(bag_box, [l["views"][0] for l in bags])
        sides = pool.map(bag_box, [l["views"][1] for l in bags])
        ratio = [(s[2] - s[0]) / (t[2] - t[0]) for t, s in zip(tops, sides) if t[2] - t[0] > 50]
        print(f"\nDvXray side/top bag length: {statistics.median(ratio):.2f} "
              f"(IQR {np.percentile(ratio, 25):.2f}-{np.percentile(ratio, 75):.2f})")


def main() -> None:
    lines = read()
    sizes = defaultdict(lambda: defaultdict(list))
    for line in lines:
        for item in line["items"]:
            if item["box"] and line["dataset"] != "compass_xp":
                view = f"/view{item['view']}" if line["dataset"] == "dvxray" else ""
                column = f"{line['dataset']}/{scan_format(line['sizes'][item['view']][1])}{view}"
                sizes[column][item["name"]].append((item["box"][2] * item["box"][3]) ** 0.5)
    stcray = defaultdict(list)
    for column, items in sizes.items():
        if column.startswith("stcray"):
            for name, values in items.items():
                stcray[name] += values

    print(f"{'dataset/format':<22} {'factor':>7}  {'item types':>10}  {'per-item range':>16}")
    for column in sorted(sizes):
        ratios = sorted(statistics.median(stcray[n]) / statistics.median(v)
                        for n, v in sizes[column].items() if len(v) >= MIN_ITEMS and len(stcray[n]) >= MIN_ITEMS)
        print(f"{column:<22} {statistics.median(ratios):>7.2f}  {len(ratios):>10}  {ratios[0]:>7.2f} to {ratios[-1]:.2f}")
    bag_rulers(lines)


if __name__ == "__main__":
    main()
