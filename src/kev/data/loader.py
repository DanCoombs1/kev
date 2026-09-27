"""Scans for training and evaluation: prepared, augmented, with threat insertion, packed into batches of patches.

    uv run python -m kev.data.loader     (builds the cached threat libraries and patch counts)
"""

import json
import zlib
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image

from kev.data.augment import orient
from kev.data.curate import read
from kev.data.insertion import Threat, cut_out, insert_randomly, load, resave_jpeg, single_threats, threat_spot, to_uint8
from kev.data.names import group
from kev.data.prepare import PATCH, SCALE, prepare_image, prepare_view
from kev.data.sources import DATA

ZOOM = 0.2
LIBRARY_SIZES = {"train": 5000, "val": 600, "test": 1000}
ROUND_TO = 64  # batch lengths come in few shapes, so the GPU reuses the kernels it built for each


@dataclass
class Library:
    threats: list[tuple[str, str, np.ndarray, np.ndarray]]  # name, source id, transmission (uint8), mask
    spots: list[tuple[float, float]]

    def threat(self, i: int) -> Threat:
        name, source, transmission, mask = self.threats[i]
        return Threat(name, transmission.astype(np.float32) / 255, mask, source, 0.0)


def random_zoom(rng: np.random.Generator, n: int | None = None):
    return np.exp(rng.uniform(np.log(1 - ZOOM), np.log(1 + ZOOM), n))


def library_path(role: str):
    return DATA / f"threats_{role}.npz"


PATCH_COUNTS = DATA / "patch_counts.json"


def _patch_count(line: dict) -> int:
    total = 0
    for view in range(len(line["views"])):
        image = prepare_view(line, view).image
        total += -(-image.width // PATCH) * -(-image.height // PATCH)
    return total


def measure_patch_counts(lines: list[dict]) -> dict[str, int]:
    usable = [l for l in lines if l["dataset"] in SCALE]
    with Pool() as pool:
        return dict(zip((l["id"] for l in usable), pool.map(_patch_count, usable, chunksize=32)))


def _cut(job: tuple) -> tuple | None:
    threat = cut_out(*job)
    if threat is None:
        return None
    return threat.name, threat.source, to_uint8(threat.transmission), threat.mask


def _spot(line: dict) -> tuple[float, float]:
    return threat_spot(line, load(line["views"][0]))


def build_library(lines: list[dict], role: str, n: int, seed: int = 0) -> Library:
    singles = single_threats(lines, role)
    rng = np.random.default_rng(seed)
    picked = [singles[i] for i in rng.choice(len(singles), min(n, len(singles)), replace=False)]
    with Pool() as pool:
        threats = [t for t in pool.map(_cut, picked, chunksize=16) if t]
        spots = pool.map(_spot, [line for line, _, _ in picked[:1000]], chunksize=16)
    return Library(threats, spots)


def save_library(library: Library, role: str) -> None:
    np.savez(
        library_path(role),
        names=np.array([t[0] for t in library.threats]),
        sources=np.array([t[1] for t in library.threats]),
        shapes=np.array([t[3].shape for t in library.threats], dtype=np.int32),
        transmission=np.concatenate([t[2].ravel() for t in library.threats]),
        mask=np.concatenate([t[3].ravel() for t in library.threats]),
        spots=np.array(library.spots, dtype=np.float32),
    )


def load_library(role: str) -> Library:
    data = np.load(library_path(role), allow_pickle=False)
    # Indexing an npz reads the whole array again, so read each one once and slice views out of it.
    all_transmission, all_mask = data["transmission"], data["mask"]
    threats, at_t, at_m = [], 0, 0
    for name, source, (h, w) in zip(data["names"], data["sources"], data["shapes"]):
        transmission = all_transmission[at_t: at_t + h * w * 3].reshape(h, w, 3)
        mask = all_mask[at_m: at_m + h * w].reshape(h, w)
        threats.append((str(name), str(source), transmission, mask))
        at_t, at_m = at_t + h * w * 3, at_m + h * w
    return Library(threats, [tuple(s) for s in data["spots"].tolist()])


class ScanDataset(torch.utils.data.Dataset):
    """One role's scans. STCray clean bags appear twice: as they are, and with an inserted threat."""

    def __init__(self, lines: list[dict], role: str, train: bool, library: Library | None = None, seed: int = 0):
        self.lines = [l for l in lines if l["role"] == role and l["dataset"] in SCALE]
        self.counts = json.loads(PATCH_COUNTS.read_text()) if PATCH_COUNTS.exists() else {}
        self.train = train
        self.library = library
        self.seed = seed
        self.entries = []
        for i, line in enumerate(self.lines):
            self.entries.append((i, False))
            if library and self.is_pair(line):
                self.entries.append((i, True))

    @staticmethod
    def is_pair(line: dict) -> bool:
        return line["dataset"] == "stcray" and not line["items"]

    def __len__(self) -> int:
        return len(self.entries)

    def size(self, k: int) -> float:
        """Patch count before augmentation, for grouping similar sizes into batches."""
        line = self.lines[self.entries[k][0]]
        if line["id"] in self.counts:
            return self.counts[line["id"]]
        return sum(w * h for w, h in line["sizes"]) * SCALE[line["dataset"]] ** 2 / PATCH**2

    def _rng(self, line: dict, inserted: bool) -> np.random.Generator:
        if self.train:
            return np.random.default_rng()
        return np.random.default_rng(zlib.crc32(f"{self.seed}:{line['id']}:{inserted}".encode()))

    def _insert(self, pixels: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict] | None:
        for _ in range(10):  # some threats don't fit in some bags
            threat = self.library.threat(int(rng.integers(len(self.library.threats))))
            result = insert_randomly(pixels, threat, rng, self.library.spots)
            if result is not None:
                composite, box = result
                item = {"name": threat.name, "group": group(threat.name, "stcray"), "label": None,
                        "view": 0, "box": box, "difficult": False}
                return composite, item
        return None

    def __getitem__(self, key: int | tuple[int, float]) -> dict:
        # BucketSampler(zoom=True) picks the zoom itself, so it can batch by zoomed size.
        k, zoom = key if isinstance(key, tuple) else (key, None)
        index, insert = self.entries[k]
        line = self.lines[index]
        rng = self._rng(line, insert)
        items = [dict(i) for i in line["items"]]
        images = [load(path) for path in line["views"]]
        inserted = False
        if self.library and self.is_pair(line):
            result = self._insert(images[0], rng) if insert else None
            if result:
                images[0], item = result
                items, inserted = [item], True
            else:
                images[0] = resave_jpeg(images[0])  # both halves of a pair go through the same compression

        if zoom is None:
            zoom = float(random_zoom(rng)) if self.train else 1.0
        two_views = len(images) > 1
        quarter_turns = 0 if two_views or not self.train else int(rng.integers(4))
        mirror = bool(rng.integers(2)) if self.train else False

        views = []
        for v, pixels in enumerate(images):
            own = [i for i in items if i["view"] == v]
            prepared = prepare_image(Image.fromarray(to_uint8(pixels)), line["dataset"], [i["box"] for i in own if i["box"]], zoom)
            image, boxes = orient(prepared.image, [prepared.box(i["box"]) if i["box"] else None for i in own], quarter_turns, mirror)
            for item, box in zip(own, boxes):
                item["box"] = box
            views.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
        return {"id": line["id"], "dataset": line["dataset"], "views": views, "items": items,
                "yes_only": line["yes_only"], "inserted": inserted, "pair": bool(self.library and self.is_pair(line))}


def to_patches(view: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """(3, h, w) -> (rows * cols, 3 * PATCH * PATCH), row by row, padding the edges with white."""
    _, h, w = view.shape
    rows, cols = -(-h // PATCH), -(-w // PATCH)
    padded = torch.full((3, rows * PATCH, cols * PATCH), 255, dtype=torch.uint8)
    padded[:, :h, :w] = view
    patches = padded.reshape(3, rows, PATCH, cols, PATCH).permute(1, 3, 0, 2, 4).reshape(rows * cols, -1)
    return patches, rows, cols


def from_patches(patches: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    return patches.reshape(rows, cols, 3, PATCH, PATCH).permute(2, 0, 3, 1, 4).reshape(3, rows * PATCH, cols * PATCH)


def collate(samples: list[dict]) -> dict:
    """Each sample becomes one sequence of patches (all its views), padded to the batch's longest sequence."""
    sequences = []
    for s in samples:
        parts = []
        for v, image in enumerate(s["views"]):
            patches, rows, cols = to_patches(image)
            row, col = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
            parts.append((patches, row.flatten(), col.flatten(), torch.full((rows * cols,), v), (rows, cols)))
        sequences.append([torch.cat([p[i] for p in parts]) for i in range(4)] + [[p[4] for p in parts]])
    length = -(-max(len(seq[0]) for seq in sequences) // ROUND_TO) * ROUND_TO
    batch = len(samples)
    patches = torch.full((batch, length, 3 * PATCH * PATCH), 255, dtype=torch.uint8)
    row, col, view = (torch.zeros((batch, length), dtype=torch.long) for _ in range(3))
    valid = torch.zeros((batch, length), dtype=torch.bool)
    for i, (p, r, c, v, _) in enumerate(sequences):
        n = len(p)
        patches[i, :n], row[i, :n], col[i, :n], view[i, :n], valid[i, :n] = p, r, c, v, True
    return {"patches": patches, "row": row, "col": col, "view": view, "valid": valid,
            "grids": [seq[4] for seq in sequences],
            "samples": [{k: v for k, v in s.items() if k != "views"} for s in samples]}


class BucketSampler(torch.utils.data.Sampler):
    """Shuffled batches of similar-sized scans, to keep padding low. With zoom, yields (index, zoom) pairs."""

    def __init__(self, sizes: list[float], batch_size: int, shuffle: bool = True, seed: int = 0, pool: int = 50,
                 zoom: bool = False):
        self.sizes = np.asarray(sizes)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.pool = pool
        self.zoom = zoom
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self.sizes)) if self.shuffle else np.arange(len(self.sizes))
        zooms = random_zoom(rng, len(self.sizes)) if self.zoom else np.ones(len(self.sizes))
        sizes = self.sizes * zooms**2
        batches = []
        step = self.pool * self.batch_size
        for start in range(0, len(order), step):
            chunk = sorted(order[start: start + step], key=lambda i: sizes[i])
            if self.zoom:
                chunk = [(int(i), float(zooms[i])) for i in chunk]
            batches += [chunk[i: i + self.batch_size] for i in range(0, len(chunk), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return -(-len(self.sizes) // self.batch_size)


def main() -> None:
    lines = read()
    counts = measure_patch_counts(lines)
    PATCH_COUNTS.write_text(json.dumps(counts))
    print(f"patch counts for {len(counts):,} scans -> {PATCH_COUNTS.name}")
    for role, n in LIBRARY_SIZES.items():
        library = build_library(lines, role, n)
        save_library(library, role)
        size = library_path(role).stat().st_size / 1e6
        print(f"{role}: {len(library.threats):,} threats, {len(library.spots)} placements -> {library_path(role).name} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
