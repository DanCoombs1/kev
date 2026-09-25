"""Contact sheets of real scans with their annotations, written to runs/explore/.

    uv run python scripts/view_scans.py
"""

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from kev.data.duplicates import DUPLICATES
from kev.data.sources import DATA, Scan, all_scans, scan_format

OUT = Path(__file__).resolve().parents[1] / "runs" / "explore"
FONT = ImageFont.load_default(size=16)
COLOURS = [(230, 0, 180), (0, 150, 255), (255, 110, 0), (0, 180, 80)]
PATCH = 16


def draw_view(scan: Scan, view: int) -> Image.Image:
    im = Image.open(scan.views[view]).convert("RGB")
    draw = ImageDraw.Draw(im)
    labels = sorted({i.label for i in scan.items})
    hidden = []
    for item in scan.items:
        if item.view != view:
            continue
        if item.box is None:
            if item.difficult:
                hidden.append(item.label)
            continue
        colour = COLOURS[labels.index(item.label) % len(COLOURS)]
        x, y, w, h = item.box
        draw.rectangle([x, y, x + w, y + h], outline=colour, width=3)
        draw.text((x + 4, y + 2), item.label, fill=colour, font=FONT)
    if hidden:
        draw.text((6, 6), "not boxed in this view: " + ", ".join(hidden), fill=(200, 0, 0), font=FONT)
    return im


def side_by_side(images: list[Image.Image], gap: int = 12) -> Image.Image:
    out = Image.new("RGB", (sum(i.width for i in images) + gap * (len(images) - 1), max(i.height for i in images)), "white")
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out


def whole_scan(scan: Scan) -> Image.Image:
    images = [draw_view(scan, v) for v in range(len(scan.views))]
    if scan.dataset == "compass_xp":
        photo = Image.open(str(scan.views[0]).replace("/Colour/", "/Photo/").replace(".png", ".jpg")).convert("RGB")
        height = images[0].height
        images.append(photo.resize((round(photo.width * height / photo.height), height)))
    return side_by_side(images)


def sheet(tiles: list[tuple[Image.Image, str]], columns: int, cell: tuple[int, int], shrink: bool = True) -> Image.Image:
    width, height = cell
    caption = 26
    rows = (len(tiles) + columns - 1) // columns
    out = Image.new("RGB", (columns * width, rows * (height + caption)), "white")
    draw = ImageDraw.Draw(out)
    for n, (im, text) in enumerate(tiles):
        x, y = (n % columns) * width, (n // columns) * (height + caption)
        if shrink:
            im = im.copy()
            im.thumbnail((width - 10, height - 10))
        draw.text((x + 6, y + 4), text, fill="black", font=FONT)
        out.paste(im, (x + 5, y + caption))
    return out


def describe(scan: Scan) -> str:
    if scan.dataset == "compass_xp":
        return f"{scan.items[0].label} ({'dangerous' if scan.meta['dangerous'] else 'harmless'})"
    items = sorted({i.label for i in scan.items})
    return f"{scan.split}: " + (", ".join(items) if items else "clean")


def crop(scan: Scan, box: tuple[float, float, float, float], view: int = 0, margin: int = 8) -> Image.Image:
    x, y, w, h = box
    with Image.open(scan.views[view]) as im:
        return im.convert("RGB").crop((int(x) - margin, int(y) - margin, int(x + w) + margin, int(y + h) + margin))


def column_of(scan: Scan) -> str:
    height = scan.sizes[0][1] if scan.sizes else 600  # DvXray views are all 800x600
    return f"{scan.dataset}/{scan_format(height)}"


def dataset_sheets(scans: list[Scan], rng: random.Random) -> None:
    by_dataset = defaultdict(list)
    for s in scans:
        by_dataset[s.dataset].append(s)
    for dataset, group in by_dataset.items():
        wide = dataset in ("dvxray", "compass_xp")
        tiles = [(whole_scan(s), describe(s)) for s in rng.sample(group, 12)]
        sheet(tiles, 2 if wide else 4, (900, 360) if wide else (450, 360)).save(OUT / f"samples_{dataset}.png")


def scale_sheet(scans: list[Scan]) -> None:
    """Median-sized example of each item per scanner, at native pixel size."""
    items = ["gun", "knife", "scissors", "lighter"]
    columns = ["pidray/medium", "stcray/short", "stcray/tall", "dvxray/medium"]
    found = defaultdict(list)
    for s in scans:
        for i in s.items:
            if i.box is not None and i.label.lower() in items and column_of(s) in columns and len(s.items) <= 2:
                found[(i.label.lower(), column_of(s))].append(((i.box[2] * i.box[3]) ** 0.5, s, i.box, i.view))
    tiles = []
    for item in items:
        for c in columns:
            candidates = found[(item, c)]
            middle = statistics.median(size for size, *_ in candidates)
            size, s, box, view = min(candidates, key=lambda t: abs(t[0] - middle))
            im = crop(s, box, view)
            ImageDraw.Draw(im).rectangle([2, 2, 2 + PATCH, 2 + PATCH], outline=(255, 0, 0), width=1)
            tiles.append((im, f"{item} · {c} · {size:.0f}px"))
    sheet(tiles, len(columns), (340, 300), shrink=False).save(OUT / "item_scale.png")


def stcray_shared_variants_sheet(scans: list[Scan], rng: random.Random) -> None:
    """STCray item variants whose names appear in both train and test."""
    by_variant = defaultdict(lambda: defaultdict(list))
    for s in scans:
        if s.dataset == "stcray" and len(s.items) == 1:
            by_variant[s.views[0].stem.split("_")[0]][s.split].append(s)
    shared = sorted(v for v, splits in by_variant.items() if splits["train"] and splits["test"])
    tiles = []
    for variant in shared:
        picks = rng.sample(by_variant[variant]["train"], 2) + rng.sample(by_variant[variant]["test"], 2)
        images = [crop(s, s.items[0].box, margin=12) for s in picks]
        tiles.append((side_by_side(images, gap=30), f"{variant}: train, train, test, test"))
    sheet(tiles, 2, (760, 250)).save(OUT / "stcray_shared_variants.png")


def unclear_labels_sheet(scans: list[Scan], rng: random.Random) -> None:
    unclear = [("dvxray", "Bat"), ("dvxray", "Pressure_vessel"), ("dvxray", "Dart"), ("dvxray", "Fireworks"),
               ("stcray", "Cutter"), ("stcray", "Blade"), ("stcray", "Other Sharp Item"), ("stcray", "Nail Cutter"),
               ("iedxray", "Modified parts"), ("pidray", "Sprayer")]
    tiles = []
    for dataset, label in unclear:
        found = [(s, i) for s in scans if s.dataset == dataset for i in s.items if i.label == label and i.box and i.view == 0]
        images = [crop(s, i.box, i.view, margin=12) for s, i in rng.sample(found, 4)]
        tiles.append((side_by_side(images, gap=24), f"{dataset}: {label}"))
    sheet(tiles, 2, (760, 280)).save(OUT / "unclear_labels.png")


def duplicate_label_conflicts_sheet(scans: list[Scan], rng: random.Random) -> None:
    """Duplicate scans whose copies are labelled differently."""
    by_id = {str(s.views[0].relative_to(DATA).with_suffix("")): s for s in scans}
    labels = lambda s: sorted(i.label for i in s.items if i.view == 0)
    tiles = []
    for dataset in ["pidray", "stcray"]:
        pairs = [(by_id[p["a"]], by_id[p["b"]]) for p in json.loads(DUPLICATES.read_text())
                 if p["a"].startswith(dataset) and p["b"].startswith(dataset)]
        differing = [(a, b) for a, b in pairs if labels(a) != labels(b)]
        for a, b in rng.sample(differing, 3):
            tiles.append((side_by_side([draw_view(a, 0), draw_view(b, 0)], gap=30),
                          f"{a.views[0].stem} {labels(a)}   vs   {b.views[0].stem} {labels(b)}"))
    sheet(tiles, 1, (1400, 420)).save(OUT / "duplicate_label_conflicts.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    scans = list(all_scans())
    duplicate_label_conflicts_sheet(scans, rng)
    dataset_sheets(scans, rng)
    scale_sheet(scans)
    stcray_shared_variants_sheet(scans, rng)
    unclear_labels_sheet(scans, rng)
    for p in sorted(OUT.glob("*.png")):
        print(p.relative_to(OUT.parents[1]))


if __name__ == "__main__":
    main()
