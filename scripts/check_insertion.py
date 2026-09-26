"""Checks for STCray threat insertion.

1. Background estimation: inpaint threat-shaped holes in clean bags and compare with the true pixels.
2. insertion_examples.png: clean bags with inserted threats, and real threat scans for comparison.
3. Same-bag pairs (clean vs clean + inserted threat): the outline probe should be near 0.5.
4. Realism: crops of a real threat and an inserted one from the same image; near 0.5 = realistic.
   Real-vs-real crops check the setup itself.

    uv run python scripts/check_insertion.py
"""

import random
import statistics
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from kev.data.curate import read
from kev.data.insertion import (CONTEXT, Threat, bag_box, bag_mask, cut_out, estimate_background, insert, insert_randomly,
                                load, place, resave_jpeg, single_threats, threat_spot, to_uint8, turn)
from kev.device import pick_device
from kev.probe import train_and_score

SIZE = 96
LIBRARY = {"train": 3000, "val": 600}
PAIRS_PER_BAG = 5
OUT = Path(__file__).resolve().parents[1] / "runs" / "explore"


def cut(job: tuple) -> Threat | None:
    return cut_out(*job)


def spot_of(line: dict) -> tuple[float, float]:
    return threat_spot(line, load(line["views"][0]))


def background_error(job: tuple) -> tuple[float, float] | None:
    path, threat, seed = job
    rng = np.random.default_rng(seed)
    pixels = load(path)
    _, mask = turn(threat, int(rng.integers(4)), bool(rng.integers(2)))
    spot = place(bag_mask(pixels), mask, rng)
    if spot is None:
        return None
    x, y = spot
    h, w = mask.shape
    x0, y0 = max(0, x - CONTEXT), max(0, y - CONTEXT)
    x1, y1 = min(pixels.shape[1], x + w + CONTEXT), min(pixels.shape[0], y + h + CONTEXT)
    hole = np.zeros(pixels.shape[:2], dtype=bool)
    hole[y:y + h, x:x + w] = mask
    crop, m = pixels[y0:y1, x0:x1], hole[y0:y1, x0:x1]
    guess = estimate_background(crop, m)
    truth = np.maximum(crop, 1 / 255)
    ring = ndimage.binary_dilation(m, iterations=6) & ~m
    return float(np.median(np.abs(guess[m] / truth[m] - 1))), float(crop[ring].std())


def square_crop(pixels: np.ndarray, box: list[float]) -> np.ndarray:
    x, y, w, h = box
    side = int(max(w, h) * 1.4) + 16
    cx, cy = int(x + w / 2), int(y + h / 2)
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    crop = to_uint8(pixels[y0:y0 + side, x0:x0 + side])
    return np.asarray(Image.fromarray(crop).resize((SIZE, SIZE), Image.Resampling.BILINEAR))


def realism_pair(job: tuple) -> tuple[np.ndarray, np.ndarray] | None:
    """Crops of the host's real threat and of an unturned threat inserted elsewhere in the same image."""
    path, real_box, threat, seed, resave, spots = job
    rng = np.random.default_rng(seed)
    pixels = load(path)
    inside = bag_mask(pixels)
    x, y, w, h = real_box
    margin = int(max(w, h) * 0.7) + 8
    inside[max(0, int(y) - margin): int(y + h) + margin, max(0, int(x) - margin): int(x + w) + margin] = False
    spot = place(inside, threat.mask, rng, spots, bag_box(pixels))
    if spot is None:
        return None
    composite = insert(pixels, threat.transmission, *spot)
    if resave:
        composite = resave_jpeg(composite)
    inserted_box = [spot[0], spot[1], threat.mask.shape[1], threat.mask.shape[0]]
    return square_crop(composite, real_box), square_crop(composite, inserted_box)


def real_pair(job: tuple) -> tuple[np.ndarray, np.ndarray]:
    (path_a, box_a), (path_b, box_b) = job
    return square_crop(load(path_a), box_a), square_crop(load(path_b), box_b)


def bag_pair(job: tuple) -> tuple[np.ndarray, np.ndarray] | None:
    """Thumbnails of a clean bag and the same bag with a threat inserted, cropped identically."""
    path, threat, seed, spots = job
    pixels = load(path)
    result = insert_randomly(pixels, threat, np.random.default_rng(seed), spots)
    if result is None:
        return None
    x0, y0, x1, y1 = bag_box(pixels)
    thumb = lambda p: np.asarray(Image.fromarray(to_uint8(p[y0:y1, x0:x1])).resize((SIZE, SIZE), Image.Resampling.BILINEAR))
    return thumb(resave_jpeg(pixels)), thumb(result[0])  # re-save the clean side too, so compression can't tell them apart


def grey(images: np.ndarray) -> np.ndarray:
    return np.repeat(images.mean(axis=3, keepdims=True).astype(np.uint8), 3, axis=3)


def outline(thumbnails: np.ndarray) -> np.ndarray:
    bag = thumbnails.mean(axis=3) < 230
    return np.repeat(np.where(bag, 0, 255).astype(np.uint8)[..., None], 3, axis=3)


def dataset(pairs: list) -> tuple[np.ndarray, np.ndarray]:
    pairs = [p for p in pairs if p is not None]
    images = np.stack([a for a, _ in pairs] + [b for _, b in pairs])
    labels = np.array([0] * len(pairs) + [1] * len(pairs), dtype=np.float32)
    return images, labels


def examples_sheet(clean: list[dict], real: list[tuple], library: list[Threat], spots: list, rng: random.Random) -> None:
    tiles = []
    for line in rng.sample(clean, 4):
        result = None
        while result is None:  # some threats don't fit in some bags
            result = insert_randomly(load(line["views"][0]), rng.choice(library), np.random.default_rng(rng.randrange(10**9)), spots)
        tiles.append(to_uint8(result[0]))
    for line, _, _ in rng.sample(real, 4):
        tiles.append(to_uint8(load(line["views"][0])))
    thumbs = []
    for im in tiles:
        im = Image.fromarray(im)
        im.thumbnail((420, 300))
        canvas = Image.new("RGB", (420, 300), "white")
        canvas.paste(im)
        thumbs.append(np.asarray(canvas))
    rows = [np.concatenate(thumbs[i: i + 4], axis=1) for i in (0, 4)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(OUT / "insertion_examples.png")


def main() -> None:
    lines = read()
    rng = random.Random(0)
    device = pick_device()
    clean = {role: [l for l in lines if l["dataset"] == "stcray" and l["role"] == role and not l["items"]] for role in LIBRARY}
    items = {role: single_threats(lines, role) for role in LIBRARY}
    hosts = {role: rng.sample(items[role], min(LIBRARY[role], len(items[role]))) for role in LIBRARY}
    with Pool() as pool:
        library, spots = {}, {}
        for role, n in LIBRARY.items():
            library[role] = [t for t in pool.map(cut, rng.sample(items[role], min(n, len(items[role]))), chunksize=16) if t]
            spots[role] = pool.map(spot_of, [line for line, _, _ in hosts[role]], chunksize=16)
            print(f"{role}: {len(clean[role])} clean bags, {len(library[role]):,} threats cut out")

        print("\n== background estimation (relative error inside threat-shaped holes in clean bags)")
        jobs = [(rng.choice(clean["train"])["views"][0], rng.choice(library["train"]), rng.randrange(10**9)) for _ in range(600)]
        errors = [e for e in pool.map(background_error, jobs, chunksize=8) if e is not None]
        cut_busyness = sorted(t.busyness for t in library["train"])
        quartiles = [cut_busyness[int(len(cut_busyness) * q)] for q in (0.25, 0.5, 0.75)]
        print(f"median {statistics.median(e for e, _ in errors):.1%}, p90 {np.percentile([e for e, _ in errors], 90):.1%}")
        for lo, hi, name in [(0, quartiles[0], "calmest quarter"), (quartiles[0], quartiles[1], "second"),
                             (quartiles[1], quartiles[2], "third"), (quartiles[2], 9, "busiest quarter")]:
            group = [e for e, b in errors if lo <= b < hi]
            if group:
                print(f"   surroundings {name:<16} median error {statistics.median(group):.1%}  ({len(group)} holes)")

        examples_sheet(clean["train"], items["train"], library["train"], spots["train"], rng)

        print("\n== same-bag pairs")
        pair_data = {}
        for role in LIBRARY:
            jobs = [(l["views"][0], rng.choice(library[role]), rng.randrange(10**9), spots[role])
                    for l in clean[role] for _ in range(PAIRS_PER_BAG)]
            pair_data[role] = dataset(pool.map(bag_pair, jobs, chunksize=8))
        (tr_x, tr_y), (va_x, va_y) = pair_data["train"], pair_data["val"]
        results = {"pairs: control": train_and_score(tr_x, tr_y, va_x, va_y, device, "control"),
                   "pairs: outline": train_and_score(outline(tr_x), tr_y, outline(va_x), va_y, device, "outline")}

        print("\n== realism")
        for variant, resave, real_spots in [("random place", False, False), ("real place + jpeg", True, True)]:
            data = {}
            for role in LIBRARY:
                jobs = [(line["views"][0], line["items"][0]["box"],
                         rng.choice([t for t in library[role][:300] if t.source != line["id"]]), rng.randrange(10**9),
                         resave, rng.sample(spots[role], 20) if real_spots else None)
                        for line, _, _ in hosts[role]]
                data[role] = dataset(pool.map(realism_pair, jobs, chunksize=8))
            (tr_x, tr_y), (va_x, va_y) = data["train"], data["val"]
            results[f"realism: {variant}"] = train_and_score(tr_x, tr_y, va_x, va_y, device, variant[:9])
            results[f"realism: {variant}, grey"] = train_and_score(grey(tr_x), tr_y, grey(va_x), va_y, device, "grey")
        Image.fromarray(np.concatenate([np.concatenate(list(va_x[:8]), axis=1),
                                        np.concatenate(list(va_x[len(va_x) // 2: len(va_x) // 2 + 8]), axis=1)])).save(OUT / "realism_crops.png")
        data = {}
        for role in LIBRARY:
            real = [(line["views"][0], line["items"][0]["box"]) for line, _, _ in hosts[role]]
            data[role] = dataset(pool.map(real_pair, list(zip(real[::2], real[1::2])), chunksize=8))
        results["realism: real vs real"] = train_and_score(*data["train"], *data["val"], device, "real/real")

    print()
    for name, value in results.items():
        print(f"{name:<32} {value:.2f}")


if __name__ == "__main__":
    main()
