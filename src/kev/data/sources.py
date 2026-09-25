"""Read each X-ray dataset's own annotation format into one common shape.

Every dataset stores its labels differently: big COCO files (PIDray, IEDXray), one
LabelMe file per image (STCray), a small JSON per bag with a box in each of two views
(DvXray), or a table of single objects (COMPASS-XP). The loaders here hide that, and
each yields `Scan`s. Labels keep the dataset's own spelling; mapping them to kev's
item names comes later.
"""

import csv
import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

DATA = Path(__file__).resolve().parents[3] / "data"  # the repo's data/ folder

Box = tuple[float, float, float, float]  # x, y, width, height in pixels


@dataclass
class Item:
    label: str  # as the dataset spells it, e.g. "Pressure_vessel"
    view: int = 0  # index into Scan.views of the image the box is drawn on
    box: Box | None = None  # None when the dataset gives no box (COMPASS-XP, or "difficult" below)
    difficult: bool = False  # DvXray: in the bag, but too hidden in this view for annotators to box


@dataclass
class Scan:
    dataset: str
    split: str
    views: list[Path]  # one image per view of the bag; DvXray has two
    sizes: list[tuple[int, int]] | None  # (width, height) per view, if the annotations record it
    items: list[Item] = field(default_factory=list)  # empty for a clean bag
    meta: dict = field(default_factory=dict)  # dataset-specific extras


def scan_format(height: int) -> str:
    """Name a scan format by its height, so e.g. PIDray's 448 px and 1040 px scans stay apart."""
    if height < 400:
        return "short"
    if height < 700:
        return "medium"
    return "tall"


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) from the file header, without decoding the pixels."""
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


def stcray(include_augmented: bool = False) -> Iterator[Scan]:
    """STCray keeps one folder per item type, and one box file per image (none for clean bags)."""
    root = DATA / "stcray"
    splits = [("train", "STCray_TrainSet", "Images", "Json_BB"), ("test", "STCray_TestSet", "Images", "Json_BB")]
    if include_augmented:  # 631k images of only 22 physical items; see Step 1a
        splits.append(("augmented", "STCray_Augmented", "Threat_Items_Images", "Threat_Items_Json_BB"))
    for split, folder, image_dir, box_dir in splits:
        for type_folder in sorted((root / folder / image_dir).iterdir()):
            if not type_folder.is_dir():
                continue
            for image in sorted(type_folder.iterdir()):
                if image.suffix.lower() not in (".jpg", ".png"):
                    continue  # stray files such as desktop.ini
                boxes = root / folder / box_dir / type_folder.name / f"{image.stem}.json"
                meta = {"folder": type_folder.name}
                if not boxes.exists():  # only the "Non Threat" images have no box file
                    yield Scan("stcray", split, [image], None, [], meta)
                    continue
                d = json.loads(boxes.read_text())
                items = [Item(s["label"], box=_corners_to_box(*s["points"][0], *s["points"][1])) for s in d["shapes"]]
                yield Scan("stcray", split, [image], [(d["imageWidth"], d["imageHeight"])], items, meta)


def dvxray() -> Iterator[Scan]:
    """Every DvXray bag has a top view (_OL) and a side view (_SD), and each item is boxed in both."""
    root = DATA / "dvxray"
    for folder in ["DvXray_Positive_Samples", "DvXray_Negative_Samples"]:
        for ann in sorted((root / folder).glob("*.json")):
            d = json.loads(ann.read_text())
            objects = [] if d["objects"] == "None" else d["objects"]  # clean bags store the string "None"
            items = []
            for o in objects:
                for view, key in enumerate(["ol_bb", "sd_bb"]):
                    if o[key] == "difficult":  # 409 boxes are this word instead of coordinates
                        items.append(Item(o["label"], view=view, difficult=True))
                    else:
                        items.append(Item(o["label"], view=view, box=_corners_to_box(*o[key])))
            views = [root / folder / f"{d['name']}_OL.png", root / folder / f"{d['name']}_SD.png"]
            yield Scan("dvxray", "all", views, None, items)  # DvXray ships without a train/test split


def compass_xp() -> Iterator[Scan]:
    """Single objects, not bags: one per scan, with no box, flagged dangerous or not."""
    root = DATA / "compass-xp" / "COMPASS-XP"
    with open(root / "meta.txt") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            meta = {"dangerous": row["dangerous"] == "True", "instance": row["instance"], "pose": row["pose"]}
            yield Scan("compass_xp", "all", [root / "Colour" / f"{row['basename']}.png"], None, [Item(row["class"])], meta)


def all_scans(include_augmented: bool = False) -> Iterator[Scan]:
    yield from pidray()
    yield from stcray(include_augmented)
    yield from dvxray()
    yield from iedxray()
    yield from compass_xp()
