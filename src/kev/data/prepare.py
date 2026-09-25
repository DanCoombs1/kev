"""Crops a view to the bag, rescales it to the common scale and caps its size."""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from kev.data.sources import DATA

# Resize factors to STCray's scale, from scripts/measure_scale.py. COMPASS-XP isn't measured yet.
SCALE = {"stcray": 1.0, "iedxray": 1.0, "dvxray": 0.86, "pidray": 0.67}
PATCH = 16
MAX_PATCHES = 1024
MARGIN = 8
BACKGROUND = 230


@dataclass
class Prepared:
    image: Image.Image
    crop: tuple[int, int, int, int]  # x0, y0, x1, y1 in the original image
    scale: float

    def box(self, box: list[float]) -> list[float]:
        x0, y0 = self.crop[:2]
        x, y, w, h = box
        return [(x - x0) * self.scale, (y - y0) * self.scale, w * self.scale, h * self.scale]

    def points(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        x0, y0 = self.crop[:2]
        return [((x - x0) * self.scale, (y - y0) * self.scale) for x, y in points]


def bag_region(grey: np.ndarray) -> tuple[int, int, int, int]:
    # Opening drops thin lines such as DvXray's belt line and PIDray's dotted bottom edge.
    solid = ndimage.binary_opening(grey < BACKGROUND, structure=np.ones((5, 5)))
    rows, cols = np.nonzero(solid)
    if len(rows) == 0:
        return 0, 0, grey.shape[1], grey.shape[0]
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def crop_region(grey: np.ndarray, boxes: list[list[float]]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bag_region(grey)
    for x, y, w, h in boxes:
        x0, y0, x1, y1 = min(x0, int(x)), min(y0, int(y)), max(x1, int(np.ceil(x + w))), max(y1, int(np.ceil(y + h)))
    height, width = grey.shape
    return max(0, x0 - MARGIN), max(0, y0 - MARGIN), min(width, x1 + MARGIN), min(height, y1 + MARGIN)


def prepare_view(line: dict, view: int = 0) -> Prepared:
    if line["dataset"] not in SCALE:
        raise ValueError(f"no scale for {line['dataset']}")
    with Image.open(DATA / line["views"][view]) as im:
        rgb = im.convert("RGB")
    boxes = [i["box"] for i in line["items"] if i["view"] == view and i["box"]]
    crop = crop_region(np.asarray(rgb.convert("L")), boxes)
    width, height = crop[2] - crop[0], crop[3] - crop[1]
    scale = SCALE[line["dataset"]]
    patches = (width * scale / PATCH) * (height * scale / PATCH)
    if patches > MAX_PATCHES:
        scale *= (MAX_PATCHES / patches) ** 0.5
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return Prepared(rgb.crop(crop).resize(size, Image.Resampling.BILINEAR), crop, scale)
