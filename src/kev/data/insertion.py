"""Threat insertion: cut threats out of STCray scans and multiply them into other bags.

Scan brightness behaves like transmission, so a threat is recovered as scan / background and
inserted as bag * threat. (Adding differences instead looked far less realistic.)
"""

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from kev.data.duplicates import content_box
from kev.data.names import kev_name
from kev.data.outlines import bounding_box, polygon_mask, true_outlines
from kev.data.prepare import BACKGROUND
from kev.data.sources import DATA

CONTEXT = 12  # pixels around an outline used to inpaint the background behind it
INPAINT_RADIUS = 5
JPEG_QUALITY = 92


@dataclass
class Threat:
    name: str
    transmission: np.ndarray  # h, w, 3 in (0, 1]; 1 outside the mask
    mask: np.ndarray
    source: str
    busyness: float  # std just outside the outline; busier surroundings give a worse background estimate


def load(path: str) -> np.ndarray:
    with Image.open(DATA / path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255


def to_uint8(pixels: np.ndarray) -> np.ndarray:
    return (np.clip(pixels, 0, 1) * 255).round().astype(np.uint8)


def resave_jpeg(pixels: np.ndarray) -> np.ndarray:
    # STCray scans are JPEGs; re-saving gives inserted regions the same compression artefacts.
    buffer = io.BytesIO()
    Image.fromarray(to_uint8(pixels)).save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32) / 255


def estimate_background(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    filled = cv2.inpaint(to_uint8(pixels), mask.astype(np.uint8), INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return np.maximum(filled.astype(np.float32) / 255, 1 / 255)


def cut_out(line: dict, label: str, points: list[tuple[float, float]]) -> Threat | None:
    pixels = load(line["views"][0])
    height, width = pixels.shape[:2]
    # Hand-drawn outlines sit on the object's edge; grow them slightly so the rim comes with it.
    mask = ndimage.binary_dilation(polygon_mask(points, (width, height)), iterations=2)
    x, y, w, h = bounding_box(points)
    x0, y0 = max(0, int(x) - CONTEXT), max(0, int(y) - CONTEXT)
    x1, y1 = min(width, int(x + w) + CONTEXT + 1), min(height, int(y + h) + CONTEXT + 1)
    crop, m = pixels[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    if m.sum() < 20:
        return None
    background = estimate_background(crop, m)
    transmission = np.where(m[..., None], np.clip(crop / background, 0, 1), 1.0).astype(np.float32)
    ring = ndimage.binary_dilation(m, iterations=6) & ~m
    busyness = float(crop[ring].std()) if ring.any() else 1.0
    rows, cols = np.nonzero(m)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    return Threat(kev_name(line["dataset"], label), transmission[r0:r1, c0:c1], m[r0:r1, c0:c1], line["id"], busyness)


def turn(threat: Threat, quarter_turns: int, mirror: bool) -> tuple[np.ndarray, np.ndarray]:
    transmission, mask = np.rot90(threat.transmission, quarter_turns), np.rot90(threat.mask, quarter_turns)
    if mirror:
        transmission, mask = transmission[:, ::-1], mask[:, ::-1]
    return np.ascontiguousarray(transmission), np.ascontiguousarray(mask)


def bag_mask(pixels: np.ndarray) -> np.ndarray:
    solid = ndimage.binary_opening(pixels.mean(axis=2) * 255 < BACKGROUND, structure=np.ones((5, 5)))
    return ndimage.binary_fill_holes(solid)


def bag_box(pixels: np.ndarray) -> tuple[int, int, int, int]:
    return content_box(to_uint8(pixels).mean(axis=2))


def threat_spot(line: dict, pixels: np.ndarray) -> tuple[float, float]:
    """Where a scan's first threat sits, as fractions of its bag's width and height."""
    x, y, w, h = line["items"][0]["box"]
    x0, y0, x1, y1 = bag_box(pixels)
    return (x + w / 2 - x0) / (x1 - x0), (y + h / 2 - y0) / (y1 - y0)


def place(inside: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
          spots: list[tuple[float, float]] | None = None, bag: tuple[int, int, int, int] | None = None,
          tries: int = 100) -> tuple[int, int] | None:
    """A top-left corner where the whole threat lies over the bag.

    With `spots`, it first tries to centre the threat near places real threats occupy (fractions of
    `bag`), which looks more natural, then falls back to anywhere over the bag.
    """
    h, w = mask.shape
    height, width = inside.shape
    if h >= height or w >= width:
        return None
    for i in range(2 * tries if spots else tries):
        if spots and i < tries:
            fx, fy = spots[i % len(spots)]
            x0, y0, x1, y1 = bag
            x = int(x0 + fx * (x1 - x0) - w / 2 + rng.normal(0, 0.03) * (x1 - x0))
            y = int(y0 + fy * (y1 - y0) - h / 2 + rng.normal(0, 0.03) * (y1 - y0))
            if not (0 <= x < width - w and 0 <= y < height - h):
                continue
        else:
            x, y = int(rng.integers(0, width - w)), int(rng.integers(0, height - h))
        if inside[y:y + h, x:x + w][mask].all():
            return x, y
    return None


def insert(bag: np.ndarray, transmission: np.ndarray, x: int, y: int) -> np.ndarray:
    out = bag.copy()
    h, w = transmission.shape[:2]
    out[y:y + h, x:x + w] *= transmission
    return out


def insert_randomly(bag: np.ndarray, threat: Threat, rng: np.random.Generator,
                    spots: list[tuple[float, float]] | None = None) -> tuple[np.ndarray, list[float]] | None:
    """The bag with the threat turned, placed and inserted, re-saved as JPEG, and the threat's new box."""
    transmission, mask = turn(threat, int(rng.integers(4)), bool(rng.integers(2)))
    spot = place(bag_mask(bag), mask, rng, rng.permutation(spots).tolist() if spots else None, bag_box(bag))
    if spot is None:
        return None
    x, y = spot
    return resave_jpeg(insert(bag, transmission, x, y)), [x, y, mask.shape[1], mask.shape[0]]


def single_threats(lines: list[dict], role: str) -> list[tuple[dict, str, list[tuple[float, float]]]]:
    """STCray scans of one role holding exactly one threat with a usable outline."""
    items = []
    for line in lines:
        if line["dataset"] == "stcray" and line["role"] == role and len(line["items"]) == 1:
            outlines = true_outlines(line)
            if len(outlines) == 1 and len(outlines[0][1]) >= 3:
                items.append((line, *outlines[0]))
    return items


def remove(pixels: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray | None:
    """The scan with a threat inpainted away, or None if its outline barely touches the image."""
    height, width = pixels.shape[:2]
    mask = ndimage.binary_dilation(polygon_mask(points, (width, height)), iterations=2)
    x, y, w, h = bounding_box(points)
    x0, y0 = max(0, int(x) - CONTEXT), max(0, int(y) - CONTEXT)
    x1, y1 = min(width, int(x + w) + CONTEXT + 1), min(height, int(y + h) + CONTEXT + 1)
    out = pixels.copy()
    region, m = out[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    if m.sum() < 20:
        return None
    region[m] = estimate_background(region, m)[m]
    return out
