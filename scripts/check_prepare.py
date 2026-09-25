"""Patch counts after preparation, plus crops.png and scale_before_after.png in runs/explore/.

    uv run python scripts/check_prepare.py
"""

import random
import statistics
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from kev.data.curate import read
from kev.data.prepare import MAX_PATCHES, PATCH, SCALE, prepare_view
from kev.data.sources import DATA

OUT = Path(__file__).resolve().parents[1] / "runs" / "explore"
FONT = ImageFont.load_default(size=16)


def patches(job: tuple[dict, int]) -> float:
    line, view = job
    image = prepare_view(line, view).image
    return (image.width / PATCH) * (image.height / PATCH)


def crops_sheet(lines: list[dict], rng: random.Random) -> None:
    picks = []
    for dataset in SCALE:
        picks += [(l, 0) for l in rng.sample([l for l in lines if l["dataset"] == dataset], 3)]
    picks += [(l, 0) for l in rng.sample([l for l in lines if l["dataset"] == "pidray" and l["yes_only"]], 2)]
    picks += [(l, 1) for l in rng.sample([l for l in lines if l["dataset"] == "dvxray"], 3)]
    tiles = []
    for line, view in picks:
        im = Image.open(DATA / line["views"][view]).convert("RGB")
        x0, y0, x1, y1 = prepare_view(line, view).crop
        ImageDraw.Draw(im).rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 0, 0), width=4)
        im.thumbnail((420, 320))
        tiles.append((im, f"{line['dataset']}{' side view' if view else ''}"))
    columns = 4
    sheet = Image.new("RGB", (columns * 430, ((len(tiles) + columns - 1) // columns) * 350), "white")
    draw = ImageDraw.Draw(sheet)
    for n, (im, caption) in enumerate(tiles):
        x, y = (n % columns) * 430, (n // columns) * 350
        draw.text((x + 5, y + 4), caption, fill="black", font=FONT)
        sheet.paste(im, (x + 5, y + 26))
    sheet.save(OUT / "crops.png")


def scale_sheet(lines: list[dict]) -> None:
    items = ["gun", "knife", "scissors", "lighter"]
    found = defaultdict(list)
    for line in lines:
        if line["dataset"] in SCALE and len(line["items"]) <= 2:
            for i in line["items"]:
                if i["view"] == 0 and i["box"] and i["name"] in items:
                    found[(i["name"], line["dataset"])].append(((i["box"][2] * i["box"][3]) ** 0.5, line, i["box"]))
    datasets = [d for d in SCALE if any((it, d) in found for it in items)]
    cell = 170
    sheet = Image.new("RGB", (len(datasets) * 2 * cell + 20, len(items) * (cell + 30) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    for c, dataset in enumerate(datasets):
        draw.text((c * 2 * cell + 5, 4), f"{dataset}: before | after", fill="black", font=FONT)
    for r, item in enumerate(items):
        for c, dataset in enumerate(datasets):
            candidates = found.get((item, dataset))
            if not candidates:
                continue
            middle = statistics.median(s for s, *_ in candidates)
            _, line, box = min(candidates, key=lambda t: abs(t[0] - middle))
            with Image.open(DATA / line["views"][0]) as im:
                x, y, w, h = box
                before = im.convert("RGB").crop((int(x), int(y), int(x + w), int(y + h)))
            prepared = prepare_view(line, 0)
            x, y, w, h = prepared.box(box)
            after = prepared.image.crop((int(x), int(y), int(x + w), int(y + h)))
            top = 30 + r * (cell + 30)
            draw.text((c * 2 * cell + 5, top), item, fill="black", font=FONT)
            for k, im in enumerate([before, after]):
                sheet.paste(im.crop((0, 0, min(im.width, cell - 10), min(im.height, cell - 10))), (c * 2 * cell + k * cell + 5, top + 22))
    sheet.save(OUT / "scale_before_after.png")


def main() -> None:
    lines = [l for l in read() if l["dataset"] in SCALE]
    rng = random.Random(0)
    sample = []
    for dataset in SCALE:
        pool = [l for l in lines if l["dataset"] == dataset]
        views = 2 if dataset == "dvxray" else 1
        sample += [(l, v) for l in rng.sample(pool, 500) for v in range(views)]
    with Pool() as workers:
        counts = workers.map(patches, sample, chunksize=16)
    per = defaultdict(list)
    for (line, view), n in zip(sample, counts):
        per[f"{line['dataset']}{' side' if view else ''}"].append(n)
    print(f"patches per view (cap {MAX_PATCHES})")
    print(f"{'dataset':<14} {'median':>7} {'p95':>6} {'max':>6}  {'capped':>7}")
    for key, values in per.items():
        values.sort()
        capped = sum(v >= MAX_PATCHES - 1 for v in values) / len(values)
        print(f"{key:<14} {statistics.median(values):>7.0f} {values[int(0.95 * len(values))]:>6.0f} {values[-1]:>6.0f}  {capped:>7.1%}")
    crops_sheet(lines, rng)
    scale_sheet(lines)


if __name__ == "__main__":
    main()
