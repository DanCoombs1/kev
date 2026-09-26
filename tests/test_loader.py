import numpy as np
import pytest
import torch
from PIL import Image

from kev.data.augment import orient
from kev.data.curate import CURATED, read
from kev.data.loader import ZOOM, BucketSampler, ScanDataset, collate, from_patches, library_path, load_library, to_patches
from kev.data.prepare import PATCH


def test_orient_moves_boxes_with_the_image():
    width, height, box = 120, 80, [30, 10, 40, 25]
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[10:35, 30:70] = (255, 0, 0)
    for turns in range(4):
        for mirror in (False, True):
            image, (moved,) = orient(Image.fromarray(pixels), [box], turns, mirror)
            red = np.asarray(image)[..., 0] == 255
            x, y, w, h = (int(round(v)) for v in moved)
            assert red[y:y + h, x:x + w].all() and red.sum() == w * h, (turns, mirror, moved)


def test_patches_go_back_together_into_the_view():
    view = torch.randint(0, 255, (3, 33, 50), dtype=torch.uint8)
    patches, rows, cols = to_patches(view)
    assert patches.shape == (3 * 4, 3 * PATCH * PATCH)
    rebuilt = from_patches(patches, rows, cols)
    assert torch.equal(rebuilt[:, :33, :50], view)
    assert (rebuilt[:, 33:, :] == 255).all() and (rebuilt[:, :, 50:] == 255).all()


def test_collate_packs_each_sample_into_one_sequence():
    samples = [{"views": [torch.zeros(3, 40, 70, dtype=torch.uint8)], "items": []},
               {"views": [torch.zeros(3, 33, 20, dtype=torch.uint8), torch.zeros(3, 16, 16, dtype=torch.uint8)], "items": []}]
    batch = collate(samples)
    assert batch["patches"].shape == (2, 15, 3 * PATCH * PATCH)
    assert batch["valid"].sum(dim=1).tolist() == [15, 7]
    assert batch["view"][1, :7].tolist() == [0] * 6 + [1]
    assert batch["row"][1, :7].tolist() == [0, 0, 1, 1, 2, 2, 0] and batch["col"][1, :7].tolist() == [0, 1, 0, 1, 0, 1, 0]
    assert batch["grids"] == [[(3, 5)], [(3, 2), (1, 1)]]
    assert (batch["patches"][1, 7:] == 255).all()


def test_bucket_sampler_uses_every_scan_once_per_epoch():
    sampler = BucketSampler(list(np.random.default_rng(0).random(1000)), batch_size=32)
    for _ in range(2):
        seen = [i for batch in sampler for i in batch]
        assert sorted(seen) == list(range(1000))


def test_bucket_sampler_picks_zooms_in_range():
    sampler = BucketSampler(list(np.random.default_rng(0).random(1000)), batch_size=32, zoom=True)
    pairs = [pair for batch in sampler for pair in batch]
    assert sorted(i for i, _ in pairs) == list(range(1000))
    assert all(1 - ZOOM <= z <= 1 + ZOOM for _, z in pairs)


needs_data = pytest.mark.skipif(not CURATED.exists() or not library_path("val").exists(),
                                reason="needs data/curated.jsonl and the threat libraries")


@pytest.fixture(scope="module")
def lines():
    return read()


def boxes_inside(sample: dict) -> bool:
    for item in sample["items"]:
        if item["box"]:
            x, y, w, h = item["box"]
            _, height, width = sample["views"][item["view"]].shape
            if not (x >= -0.5 and y >= -0.5 and x + w <= width + 1 and y + h <= height + 1):
                return False
    return True


@needs_data
def test_training_samples_keep_boxes_inside_their_views(lines):
    dataset = ScanDataset(lines, "train", train=True, library=load_library("train"))
    rng = np.random.default_rng(0)
    for k in rng.choice(len(dataset), 150, replace=False):
        assert boxes_inside(dataset[int(k)]), dataset.entries[int(k)]


@needs_data
def test_clean_stcray_bags_come_in_pairs_and_validation_repeats(lines):
    dataset = ScanDataset(lines, "val", train=False, library=load_library("val"))
    pairs = [k for k, (i, insert) in enumerate(dataset.entries) if insert]
    assert pairs and all(dataset.entries[k - 1] == (dataset.entries[k][0], False) for k in pairs)
    for k in pairs[:10]:
        a, b = dataset[k], dataset[k]
        assert a["inserted"] and len(a["items"]) == 1 and a["items"][0]["group"]
        assert a["items"][0]["box"] == b["items"][0]["box"] and torch.equal(a["views"][0], b["views"][0])
        assert not dataset[k - 1]["items"] and dataset[k - 1]["pair"]
