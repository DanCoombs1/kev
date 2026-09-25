"""Finds near-duplicate scans with a difference hash (dHash) and LSH banding.

    uv run python -m kev.data.duplicates
"""

import json
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from PIL import Image

from kev.data.manifest import read
from kev.data.sources import DATA

DUPLICATES = DATA / "duplicates.json"
MAX_DIFFERENT = 7  # bits out of 256
CHUNKS = 8  # with 8 bands, any pair within 7 bits shares at least one band exactly


def content_box(grey: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(grey < 230)
    if len(rows) == 0:
        return 0, 0, grey.shape[1], grey.shape[0]
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def crop_to_content(grey: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = content_box(grey)
    return grey[y0:y1, x0:x1]


def fingerprint(path: str) -> int:
    with Image.open(DATA / path) as im:
        grey = crop_to_content(np.asarray(im.convert("L")))
    small = np.asarray(Image.fromarray(grey).resize((17, 16), Image.Resampling.BILINEAR), dtype=np.int16)
    brighter = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in brighter), 2)


def near_copies(prints: list[int]) -> list[tuple[int, int, int]]:
    buckets = defaultdict(list)
    for n, fp in enumerate(prints):
        for c in range(CHUNKS):
            buckets[(c, (fp >> (32 * c)) & 0xFFFFFFFF)].append(n)
    pairs = {}
    for members in buckets.values():
        if len(members) > 2000:  # mostly-background bands
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
    with Pool() as pool:
        prints = pool.map(fingerprint, [line["views"][0] for line in lines], chunksize=256)
    pairs = near_copies(prints)
    DUPLICATES.write_text(json.dumps([{"a": ids[i], "b": ids[j], "different": d} for i, j, d in pairs]))
    print(f"{len(pairs):,} near-copy pairs written to {DUPLICATES.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
