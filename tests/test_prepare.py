import random

import pytest

from kev.data.curate import CURATED, read
from kev.data.prepare import MAX_PATCHES, PATCH, SCALE, prepare_view

if not CURATED.exists():
    pytest.skip("curated data not built: uv run python -m kev.data.curate", allow_module_level=True)

LINES = read()
RNG = random.Random(0)
SAMPLE = [(line, view) for d in SCALE for line in RNG.sample([l for l in LINES if l["dataset"] == d and l["items"]], 40)
          for view in range(len(line["views"]))]


def test_crop_keeps_every_labelled_item():
    for line, view in SAMPLE:
        x0, y0, x1, y1 = prepare_view(line, view).crop
        for item in line["items"]:
            if item["view"] == view and item["box"]:
                x, y, w, h = item["box"]
                assert x0 <= x and y0 <= y and x + w <= x1 + 1 and y + h <= y1 + 1, (line["id"], item["name"])


def test_boxes_land_inside_the_prepared_image():
    for line, view in SAMPLE:
        prepared = prepare_view(line, view)
        for item in line["items"]:
            if item["view"] == view and item["box"]:
                x, y, w, h = prepared.box(item["box"])
                assert x >= -0.5 and y >= -0.5, (line["id"], item["name"])
                assert x + w <= prepared.image.width + 1 and y + h <= prepared.image.height + 1, (line["id"], item["name"])


def test_scale_is_the_datasets_unless_capped():
    for line, view in SAMPLE:
        prepared = prepare_view(line, view)
        patches = prepared.image.width * prepared.image.height / PATCH**2
        assert patches <= MAX_PATCHES * 1.01
        if prepared.scale < SCALE[line["dataset"]] - 1e-9:
            x0, y0, x1, y1 = prepared.crop
            assert (x1 - x0) * (y1 - y0) * SCALE[line["dataset"]] ** 2 / PATCH**2 > MAX_PATCHES
        else:
            assert prepared.scale == pytest.approx(SCALE[line["dataset"]])


def test_preparing_twice_gives_the_same_result():
    line, view = SAMPLE[0]
    a, b = prepare_view(line, view), prepare_view(line, view)
    assert a.crop == b.crop and a.scale == b.scale and a.image.tobytes() == b.image.tobytes()


def test_compass_xp_is_refused():
    compass = next(l for l in LINES if l["dataset"] == "compass_xp")
    with pytest.raises(ValueError):
        prepare_view(compass)
