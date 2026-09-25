"""Check the built manifest (data/manifest.jsonl) is complete and internally consistent."""

from collections import Counter

import pytest

from kev.data.manifest import MANIFEST, TIERS, read
from kev.data.names import ALLOWED, RESTRICTED, WEAPON
from kev.data.sources import DATA

if not MANIFEST.exists():
    pytest.skip("build the manifest first: uv run python -m kev.data.manifest", allow_module_level=True)

LINES = read()


def test_ids_are_unique():
    assert len({line["id"] for line in LINES}) == len(LINES)


def test_every_image_exists():
    missing = [v for line in LINES for v in line["views"] if not (DATA / v).exists()]
    assert not missing, missing[:5]


def test_boxes_lie_inside_their_image():
    for line in LINES:
        for item in line["items"]:
            if item["box"]:
                x, y, w, h = item["box"]
                width, height = line["sizes"][item["view"]]
                assert 0 <= x and 0 <= y and x + w <= width and y + h <= height and w > 0 and h > 0, (line["id"], item)


def test_every_item_is_named_and_grouped():
    for line in LINES:
        for item in line["items"]:
            assert item["name"] and item["group"] in (WEAPON, RESTRICTED, ALLOWED), (line["id"], item)


def test_pidray_has_no_clean_bags():
    # Every PIDray bag holds a threat; a PIDray scan without items would be unlabelled, not clean.
    assert all(line["items"] for line in LINES if line["dataset"] == "pidray")


def test_licence_tiers():
    assert {line["dataset"] for line in LINES if line["tier"] == "commercial"} == {d for d, (t, _) in TIERS.items() if t == "commercial"}
    assert Counter(line["tier"] for line in LINES).keys() == {"commercial", "academic"}
