"""Item outlines: hand-drawn ones from STCray and PIDray, and two ways of estimating one from a box."""

import json
from functools import cache

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from kev.data.sources import DATA

Points = list[tuple[float, float]]


@cache
def _pidray_outlines(split: str) -> dict[str, list[tuple[str, Points]]]:
    data = json.loads((DATA / "pidray" / "annotations" / f"xray_{split}.json").read_text())
    names = {c["id"]: c["name"] for c in data["categories"]}
    files = {img["id"]: img["file_name"] for img in data["images"]}
    out = {}
    for a in data["annotations"]:
        for flat in a["segmentation"]:
            out.setdefault(files[a["image_id"]], []).append((names[a["category_id"]], list(zip(flat[::2], flat[1::2]))))
    return out


def true_outlines(line: dict) -> list[tuple[str, Points]]:
    path = DATA / line["views"][0]
    if line["dataset"] == "stcray":
        outline_file = path.parents[2] / "Json" / path.parent.name / f"{path.stem}.json"
        if not outline_file.exists():
            return []
        shapes = json.loads(outline_file.read_text())["shapes"]
        return [(s["label"], [tuple(p) for p in s["points"]]) for s in shapes if s["shape_type"] == "polygon"]
    if line["dataset"] == "pidray":
        return _pidray_outlines(line["split"]).get(path.name, [])
    return []


def polygon_mask(points: Points, size: tuple[int, int]) -> np.ndarray:
    canvas = Image.new("L", size, 0)
    ImageDraw.Draw(canvas).polygon([tuple(p) for p in points], fill=1)
    return np.asarray(canvas, dtype=bool)


def bounding_box(points: Points) -> list[float]:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def _region(box: list[float], size: tuple[int, int], margin: int) -> tuple[int, int, int, int] | None:
    x, y, w, h = box
    width, height = size
    x0, y0 = max(0, int(x) - margin), max(0, int(y) - margin)
    x1, y1 = min(width, int(x + w) + margin + 1), min(height, int(y + h) + margin + 1)
    return (x0, y0, x1, y1) if x1 - x0 >= 3 and y1 - y0 >= 3 else None  # some outlines run off the image


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    blobs, count = ndimage.label(mask)
    if count == 0:
        return mask
    biggest = 1 + int(np.argmax(ndimage.sum(mask, blobs, range(1, count + 1))))
    return ndimage.binary_fill_holes(blobs == biggest)


def ratio_outline(pixels: np.ndarray, box: list[float], darker: float = 0.85, margin: int = 4) -> np.ndarray:
    """Pixels noticeably darker than a background interpolated from the box's edges."""
    mask = np.zeros(pixels.shape[:2], dtype=bool)
    region = _region(box, (pixels.shape[1], pixels.shape[0]), margin)
    if region is None:
        return mask
    x0, y0, x1, y1 = region
    grey = pixels[y0:y1, x0:x1].mean(axis=2) + 1
    height, width = grey.shape
    down = np.linspace(0, 1, height)[:, None]
    across = np.linspace(0, 1, width)[None, :]
    background = ((1 - down) * grey[0][None, :] + down * grey[-1][None, :]
                  + (1 - across) * grey[:, 0][:, None] + across * grey[:, -1][:, None]) / 2
    mask[y0:y1, x0:x1] = _largest_blob(grey / background < darker)
    return mask


def grabcut_outline(pixels: np.ndarray, box: list[float], margin: int = 100_000, rounds: int = 5) -> np.ndarray:
    # GrabCut models the background from outside the box, so a tight crop leaves it too little
    # to go on and it often returns an empty mask. Hence the whole image by default.
    mask = np.zeros(pixels.shape[:2], dtype=bool)
    region = _region(box, (pixels.shape[1], pixels.shape[0]), margin)
    if region is None:
        return mask
    x0, y0, x1, y1 = region
    crop = np.ascontiguousarray(pixels[y0:y1, x0:x1, ::-1])
    x, y, w, h = box
    rect = (int(x) - x0, int(y) - y0, max(1, int(w)), max(1, int(h)))
    labels = np.zeros(crop.shape[:2], np.uint8)
    try:
        cv2.grabCut(crop, labels, rect, np.zeros((1, 65)), np.zeros((1, 65)), rounds, cv2.GC_INIT_WITH_RECT)
        inside = (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)
    except cv2.error:
        inside = np.zeros(crop.shape[:2], dtype=bool)
        inside[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]] = True
    mask[y0:y1, x0:x1] = _largest_blob(inside)
    return mask


def box_outline(pixels: np.ndarray, box: list[float]) -> np.ndarray:
    x, y, w, h = box
    mask = np.zeros(pixels.shape[:2], dtype=bool)
    mask[int(y): int(y + h) + 1, int(x): int(x + w) + 1] = True
    return mask
