"""Does learning "threat bag?" from insertion pairs carry over to real threats?

Trains the probe four ways on STCray train bags:
    A  clean bags vs the same bags with an inserted threat
    B  clean bags vs half inserted, half real threat bags
    C  clean bags vs real threat bags
    D  real threat bags with the threat inpainted away vs the untouched scans (a ceiling for T3)
and scores each on val:
    T1 real clean vs real threat bags (bag shape can help here)
    T2 clean bag vs the same bag with an inserted threat
    T3 real threat bag with its threat inpainted away vs the untouched scan
    E1 clean bag vs the same bag with an empty insertion (plain bag cut out and inserted like a threat)
    E2 real threat bag with an empty region inpainted vs the untouched scan
The recipe is chosen on T3, provided T2 is also good; T1 is reported only. E1 and E2 should be
near 0.5: above it, a model is reacting to the insertion or removal process rather than threats.

    uv run python scripts/insertion_experiment.py
"""

import random
from multiprocessing import Pool

import numpy as np
from PIL import Image

from kev.data.curate import read
from kev.data.insertion import (Threat, bag_box, bag_mask, cut_out, insert_randomly, load, place, remove, resave_jpeg,
                                single_threats, threat_spot, to_uint8)
from kev.data.outlines import bounding_box
from kev.device import pick_device
from kev.probe import score_auc, train

SIZE = 192
INSERTIONS_PER_BAG = 10
LIBRARY = {"train": 3000, "val": 600}


def thumb(pixels: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.asarray(Image.fromarray(to_uint8(pixels[y0:y1, x0:x1])).resize((SIZE, SIZE), Image.Resampling.BILINEAR))


def as_is(path: str) -> np.ndarray:
    pixels = load(path)
    return thumb(resave_jpeg(pixels), bag_box(pixels))


def inserted(job: tuple) -> np.ndarray | None:
    path, threat, seed, spots = job
    pixels = load(path)
    result = insert_randomly(pixels, threat, np.random.default_rng(seed), spots)
    return None if result is None else thumb(result[0], bag_box(pixels))


def removal_pair(job: tuple) -> tuple[np.ndarray, np.ndarray] | None:
    path, points = job
    pixels = load(path)
    removed = remove(pixels, points)
    if removed is None:
        return None
    box = bag_box(pixels)
    return thumb(resave_jpeg(removed), box), thumb(resave_jpeg(pixels), box)


def cut(job: tuple) -> Threat | None:
    return cut_out(*job)


def moved_away(line: dict, points: list, pixels: np.ndarray, rng: np.random.Generator) -> list | None:
    """The outline shifted to a spot over the bag, clear of the scan's real threat."""
    inside = bag_mask(pixels)
    x, y, w, h = line["items"][0]["box"]
    margin = int(max(w, h) * 0.3) + 8
    inside[max(0, int(y) - margin): int(y + h) + margin, max(0, int(x) - margin): int(x + w) + margin] = False
    bx, by, bw, bh = bounding_box(points)
    spot = place(inside, np.ones((int(bh) + 1, int(bw) + 1), dtype=bool), rng)
    if spot is None:
        return None
    return [(px + spot[0] - bx, py + spot[1] - by) for px, py in points]


def empty_threat(job: tuple) -> Threat | None:
    line, label, points, seed = job
    moved = moved_away(line, points, load(line["views"][0]), np.random.default_rng(seed))
    return None if moved is None else cut_out(line, label, moved)


def empty_removal_pair(job: tuple) -> tuple[np.ndarray, np.ndarray] | None:
    line, points, seed = job
    pixels = load(line["views"][0])
    moved = moved_away(line, points, pixels, np.random.default_rng(seed))
    removed = None if moved is None else remove(pixels, moved)
    if removed is None:
        return None
    box = bag_box(pixels)
    return thumb(resave_jpeg(removed), box), thumb(resave_jpeg(pixels), box)


def spot_of(line: dict) -> tuple[float, float]:
    return threat_spot(line, load(line["views"][0]))


def labelled(negatives: list[np.ndarray], positives: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return (np.stack(negatives + positives),
            np.array([0] * len(negatives) + [1] * len(positives), dtype=np.float32))


def main() -> None:
    lines = read()
    rng = random.Random(0)
    device = pick_device()
    stcray = [l for l in lines if l["dataset"] == "stcray"]
    clean = {role: [l for l in stcray if l["role"] == role and not l["items"]] for role in LIBRARY}
    threat_bags = {role: [l for l in stcray if l["role"] == role and l["items"]] for role in LIBRARY}
    singles = {role: single_threats(lines, role) for role in LIBRARY}

    with Pool() as pool:
        library, spots = {}, {}
        for role, n in LIBRARY.items():
            library[role] = [t for t in pool.map(cut, rng.sample(singles[role], n), chunksize=16) if t]
            spots[role] = pool.map(spot_of, [l for l, _, _ in rng.sample(singles[role], 500)], chunksize=16)

        images = {}
        for role in LIBRARY:
            n = len(clean[role]) * INSERTIONS_PER_BAG
            clean_once = pool.map(as_is, [l["views"][0] for l in clean[role]], chunksize=8)
            jobs = [(l["views"][0], rng.choice(library[role]), rng.randrange(10**9), spots[role])
                    for l in clean[role] for _ in range(INSERTIONS_PER_BAG)]
            results = pool.map(inserted, jobs, chunksize=8)
            images[role] = {
                "clean": [clean_once[i // INSERTIONS_PER_BAG] for i, r in enumerate(results) if r is not None],
                "inserted": [r for r in results if r is not None],
                "real": pool.map(as_is, [l["views"][0] for l in rng.sample(threat_bags[role], n)], chunksize=8),
            }
            print(f"{role}: {len(clean[role])} clean bags -> {len(images[role]['inserted']):,} insertions, "
                  f"{len(images[role]['real']):,} real threat bags")
        removal = [r for r in pool.map(removal_pair, [(l["views"][0], p) for l, _, p in rng.sample(singles["val"], 920)], chunksize=8) if r]
        train_removal = [r for r in pool.map(removal_pair, [(l["views"][0], p) for l, _, p in rng.sample(singles["train"], 4800)],
                                             chunksize=8) if r]
        empties = [t for t in pool.map(empty_threat, [(l, lab, p, rng.randrange(10**9)) for l, lab, p in rng.sample(singles["val"], 600)],
                                       chunksize=8) if t]
        jobs = [(l["views"][0], rng.choice(empties), rng.randrange(10**9), spots["val"])
                for l in clean["val"] for _ in range(INSERTIONS_PER_BAG)]
        empty_inserted = pool.map(inserted, jobs, chunksize=8)
        empty_removal = [r for r in pool.map(empty_removal_pair, [(l, p, rng.randrange(10**9)) for l, _, p in rng.sample(singles["val"], 920)],
                                             chunksize=8) if r]
        clean_once = pool.map(as_is, [l["views"][0] for l in clean["val"]], chunksize=8)

    tr = images["train"]
    half = len(tr["inserted"]) // 2
    recipes = {
        "A: insertion pairs": labelled(tr["clean"], tr["inserted"]),
        "B: pairs + real": labelled(tr["clean"], tr["inserted"][:half] + tr["real"][:len(tr["inserted"]) - half]),
        "C: real only": labelled(tr["clean"], tr["real"][:len(tr["inserted"])]),
        "D: removal pairs": labelled([a for a, _ in train_removal], [b for _, b in train_removal]),
    }
    va = images["val"]
    tests = {
        "T1 real bags": labelled(clean_once, va["real"]),
        "T2 insertion pairs": labelled(va["clean"], va["inserted"]),
        "T3 removal pairs": labelled([a for a, _ in removal], [b for _, b in removal]),
        "E1 empty insertion": labelled([clean_once[i // INSERTIONS_PER_BAG] for i, r in enumerate(empty_inserted) if r is not None],
                                       [r for r in empty_inserted if r is not None]),
        "E2 empty removal": labelled([a for a, _ in empty_removal], [b for _, b in empty_removal]),
    }

    results = {}
    for name, (x, y) in recipes.items():
        print(f"\ntraining {name} on {len(y):,} images")
        model = train(x, y, device, epochs=12, report=tests["T3 removal pairs"], label=name[:1])
        results[name] = {test: score_auc(model, tx, ty, device) for test, (tx, ty) in tests.items()}

    print(f"\n{'':<20}" + "".join(f"{t:>20}" for t in tests))
    for name, row in results.items():
        print(f"{name:<20}" + "".join(f"{v:>20.2f}" for v in row.values()))


if __name__ == "__main__":
    main()
