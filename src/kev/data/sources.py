"""Loaders that read each dataset's annotation format into `Scan`s. Labels keep each dataset's spelling."""

import csv
import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

DATA = Path(__file__).resolve().parents[3] / "data"

Box = tuple[float, float, float, float]  # x, y, w, h


@dataclass
class Item:
    label: str
    view: int = 0
    box: Box | None = None
    difficult: bool = False  # DvXray: present, but annotators couldn't box it in this view


@dataclass
class Scan:
    dataset: str
    split: str
    views: list[Path]
    sizes: list[tuple[int, int]] | None  # (w, h) per view, when the annotations include it
    items: list[Item] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def scan_format(height: int) -> str:
    if height < 400:
        return "short"
    if height < 700:
        return "medium"
    return "tall"


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def _corners_to_box(x1: float, y1: float, x2: float, y2: float) -> Box:
    return min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)


def _coco(annotations: Path, images: Path, dataset: str, split: str) -> Iterator[Scan]:
    data = json.loads(annotations.read_text())
    names = {c["id"]: c["name"] for c in data["categories"]}
    items = defaultdict(list)
    for a in data["annotations"]:
        items[a["image_id"]].append(Item(names[a["category_id"]], box=tuple(a["bbox"])))
    for img in data["images"]:
        meta = {"caption": img["caption"]} if "caption" in img else {}
        yield Scan(dataset, split, [images / img["file_name"]], [(img["width"], img["height"])], items[img["id"]], meta)


def pidray() -> Iterator[Scan]:
    root = DATA / "pidray"
    for split, folder in [("train", "train"), ("test_easy", "easy"), ("test_hard", "hard"), ("test_hidden", "hidden")]:
        yield from _coco(root / "annotations" / f"xray_{split}.json", root / folder, "pidray", split)


def iedxray() -> Iterator[Scan]:
    root = DATA / "iedxray" / "IEDXray"
    for split, folder in [("train", "Train"), ("test", "Test")]:
        yield from _coco(root / "annotations" / f"complete_{split}.json", root / folder, "iedxray", split)


def stcray() -> Iterator[Scan]:
    root = DATA / "stcray"
    for split, folder, image_dir, box_dir in [("train", "STCray_TrainSet", "Images", "Json_BB"),
                                              ("test", "STCray_TestSet", "Images", "Json_BB")]:
        for type_folder in sorted((root / folder / image_dir).iterdir()):
            if not type_folder.is_dir():
                continue
            for image in sorted(type_folder.iterdir()):
                if image.suffix.lower() not in (".jpg", ".png"):
                    continue
                boxes = root / folder / box_dir / type_folder.name / f"{image.stem}.json"
                meta = {"folder": type_folder.name}
                if not boxes.exists():  # clean bags have no box file
                    yield Scan("stcray", split, [image], None, [], meta)
                    continue
                d = json.loads(boxes.read_text())
                items = [Item(s["label"], box=_corners_to_box(*s["points"][0], *s["points"][1])) for s in d["shapes"]]
                yield Scan("stcray", split, [image], [(d["imageWidth"], d["imageHeight"])], items, meta)


def dvxray() -> Iterator[Scan]:
    root = DATA / "dvxray"
    for folder in ["DvXray_Positive_Samples", "DvXray_Negative_Samples"]:
        for ann in sorted((root / folder).glob("*.json")):
            d = json.loads(ann.read_text())
            objects = [] if d["objects"] == "None" else d["objects"]
            items = []
            for o in objects:
                for view, key in enumerate(["ol_bb", "sd_bb"]):
                    if o[key] == "difficult":
                        items.append(Item(o["label"], view=view, difficult=True))
                    else:
                        items.append(Item(o["label"], view=view, box=_corners_to_box(*o[key])))
            views = [root / folder / f"{d['name']}_OL.png", root / folder / f"{d['name']}_SD.png"]
            yield Scan("dvxray", "all", views, None, items)


def compass_xp() -> Iterator[Scan]:
    root = DATA / "compass-xp" / "COMPASS-XP"
    with open(root / "meta.txt") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            meta = {"dangerous": row["dangerous"] == "True", "instance": row["instance"], "pose": row["pose"],
                    "tray": row["scan_tray"]}
            yield Scan("compass_xp", "all", [root / "Colour" / f"{row['basename']}.png"], None, [Item(row["class"])], meta)


def all_scans() -> Iterator[Scan]:
    yield from pidray()
    yield from stcray()
    yield from dvxray()
    yield from iedxray()
    yield from compass_xp()
