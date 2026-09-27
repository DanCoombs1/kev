import torch

from kev.data.loader import ROUND_TO, collate
from kev.model.image import ImageEncoder
from kev.pretrain.image import (MIN_PASSES, PATIENCE, MaskedAutoencoder, choose_shown, masked_loss, passes_since_gain,
                                redraw, should_stop, target)

torch.manual_seed(0)


def batch() -> dict:
    views = [[torch.randint(0, 256, (3, 160, 176), dtype=torch.uint8)],
             [torch.randint(0, 256, (3, 64, 80), dtype=torch.uint8), torch.randint(0, 256, (3, 48, 48), dtype=torch.uint8)]]
    return collate([{"views": v, "items": []} for v in views])


def test_a_quarter_of_the_real_patches_are_shown_and_never_padding():
    b = batch()
    index, shown = choose_shown(b["valid"], torch.Generator().manual_seed(0))
    assert index.shape[1] % ROUND_TO == 0 or index.shape[1] == b["valid"].shape[1]
    for i, real in enumerate(b["valid"].sum(dim=1).tolist()):
        picked = index[i][shown[i]]
        assert len(picked) == -(-real // 4) and len(set(picked.tolist())) == len(picked)
        assert b["valid"][i, picked].all()


def test_hidden_pixels_never_reach_the_model():
    model, b = MaskedAutoencoder(ImageEncoder(layers=1)), batch()
    index, shown = choose_shown(b["valid"], torch.Generator().manual_seed(0))
    predicted, was_shown = model(b, index, shown)
    changed = dict(b, patches=torch.where((b["valid"] & ~was_shown)[..., None], 255 - b["patches"], b["patches"]))
    assert torch.allclose(model(changed, index, shown)[0], predicted, atol=1e-5)


def test_only_hidden_patches_count():
    b = batch()
    hidden = b["valid"].clone()
    hidden[:, :5] = False
    predicted = torch.randn(*b["patches"].shape)
    moved = predicted.clone()
    moved[:, :5] += 10
    assert torch.isclose(masked_loss(predicted, b["patches"], hidden), masked_loss(moved, b["patches"], hidden))


def test_targets_are_patterns_and_redraw_back():
    patches = torch.randint(0, 256, (4, 768), dtype=torch.uint8)
    patches[0] = 255  # a blank patch
    t = target(patches)
    assert torch.allclose(t[0], torch.zeros(768)) and torch.allclose(t.mean(dim=-1), torch.zeros(4), atol=1e-5)
    assert (redraw(t, patches).int() - patches.int()).abs().max() <= 1


def test_early_stopping_counts_passes_since_a_real_gain():
    assert passes_since_gain([0.5]) == 0
    assert passes_since_gain([0.5, 0.45, 0.40]) == 0
    assert passes_since_gain([0.5, 0.40, 0.3995, 0.3990, 0.3985]) == 3  # wobbles under the minimum gain don't count
    assert passes_since_gain([0.5, 0.40, 0.3990, 0.3985, 0.3975]) == 0  # small gains that add up do
    assert passes_since_gain([0.5, 0.40] + [0.41] * PATIENCE) == PATIENCE


def test_never_stops_before_the_minimum_number_of_passes():
    flat_early = [0.8, 0.45] + [0.45] * PATIENCE
    assert passes_since_gain(flat_early) >= PATIENCE and not should_stop(flat_early)
    flat_late = [0.8] + [0.8 - 0.01 * n for n in range(1, MIN_PASSES - PATIENCE + 1)]
    flat_late += [flat_late[-1]] * PATIENCE
    assert len(flat_late) - 1 == MIN_PASSES and should_stop(flat_late)
