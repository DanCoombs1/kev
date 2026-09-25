import json

import pytest

from kev.data.curate import CURATED, SAME_PICTURE, read
from kev.data.duplicates import DUPLICATES
from kev.data.manifest import read as read_manifest
from kev.data.sources import scan_format

if not CURATED.exists():
    pytest.skip("curated data not built: uv run python -m kev.data.curate", allow_module_level=True)

CURATED_LINES = {line["id"]: line for line in read()}
PUBLISHED = {line["id"]: line for line in read_manifest()}
COPIES = [p for p in json.loads(DUPLICATES.read_text()) if p["different"] <= SAME_PICTURE]


def test_no_two_surviving_scans_are_copies():
    both = [p for p in COPIES if p["a"] in CURATED_LINES and p["b"] in CURATED_LINES]
    assert not both, both[:3]


def test_every_published_scan_survives_or_is_a_listed_copy():
    listed = {c for line in CURATED_LINES.values() for c in line["copies"]}
    for i in PUBLISHED:
        assert (i in CURATED_LINES) != (i in listed), i


def test_survivors_carry_every_copys_item_names():
    for line in CURATED_LINES.values():
        names = {(i["name"], i["view"]) for i in line["items"]}
        for c in line["copies"]:
            copy = PUBLISHED[c]
            if copy["sizes"] == line["sizes"]:
                assert {(i["name"], i["view"]) for i in copy["items"]} <= names, (line["id"], c)


def test_a_test_copy_is_never_given_up_for_a_train_copy():
    for line in CURATED_LINES.values():
        if not line["split"].startswith("test"):
            assert not any(PUBLISHED[c]["split"].startswith("test") for c in line["copies"]), line["id"]


def test_only_pidray_trays_are_yes_only():
    for line in CURATED_LINES.values():
        tray = line["dataset"] == "pidray" and scan_format(line["sizes"][0][1]) == "tall"
        assert line["yes_only"] == tray, line["id"]
