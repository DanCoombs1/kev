"""Find near-copies of the same scan, anywhere in the data, with a perceptual hash.

A perceptual hash is a fingerprint of what an image looks like: crop away the empty background,
shrink to a 17x16 greyscale thumbnail, and record for each pixel whether it's brighter than its
right-hand neighbour. That's 256 yes/no answers. Near-identical images give near-identical
fingerprints, so we count how many answers differ (the Hamming distance).

Comparing every pair of ~130k scans would be 8 billion comparisons. Instead each fingerprint is
cut into 8 chunks of 32 answers, and only scans that match exactly on a chunk are compared. Two
fingerprints differing in at most 7 answers must share an untouched chunk, so no such pair is missed.

    uv run python -m kev.data.duplicates     (writes data/duplicates.json)
"""

import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from PIL import Image

from kev.data.manifest import read
from kev.data.sources import DATA

DUPLICATES = DATA / "duplicates.json"
MAX_DIFFERENT = 7  # answers out of 256 that may differ for two scans to count as near-copies
CHUNKS = 8


def content_box(grey: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) around everything darker than near-white background: roughly, the bag."""
    rows, cols = np.nonzero(grey < 230)
    if len(rows) == 0:
        return 0, 0, grey.shape[1], grey.shape[0]
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def crop_to_content(grey: np.ndarray) -> np.ndarray:
    """Cut away the near-white background around the bag."""
    x0, y0, x1, y1 = content_box(grey)
    return grey[y0:y1, x0:x1]


def fingerprint(path: str) -> int:
    with Image.open(DATA / path) as im:
        grey = crop_to_content(np.asarray(im.convert("L")))
    small = np.asarray(Image.fromarray(grey).resize((17, 16), Image.Resampling.BILINEAR), dtype=np.int16)
    brighter = (small[:, 1:] > small[:, :-1]).flatten()  # 16 x 16 = 256 answers
    return int("".join("1" if b else "0" for b in brighter), 2)


def near_copies(ids: list[str], prints: list[int]) -> list[tuple[int, int, int]]:
    """Pairs (i, j, answers that differ) of fingerprints within MAX_DIFFERENT of each other."""
    buckets = defaultdict(list)
    for n, fp in enumerate(prints):
        for c in range(CHUNKS):
            buckets[(c, (fp >> (32 * c)) & 0xFFFFFFFF)].append(n)
    pairs = {}
    for members in buckets.values():
        if len(members) > 2000:  # a chunk shared by thousands of scans is almost all background; skip it
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                if (i, j) not in pairs:
                    different = (prints[i] ^ prints[j]).bit_count()
                    if different <= MAX_DIFFERENT:
                        pairs[(i, j)] = different
    return [(i, j, d) for (i, j), d in pairs.items()]


def main() -> None:
    lines = read()
    ids = [line["id"] for line in lines]
    with Pool() as pool:  # use every CPU core: fingerprinting ~130k images is the slow part
        prints = pool.map(fingerprint, [line["views"][0] for line in lines], chunksize=256)
    pairs = near_copies(ids, prints)
    DUPLICATES.write_text(json.dumps([{"a": ids[i], "b": ids[j], "different": d} for i, j, d in pairs]))
    print(f"{len(pairs):,} near-copy pairs written to {DUPLICATES.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
