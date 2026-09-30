import torch
import torch.nn.functional as F

from kev.data.loader import collate
from kev.model.blocks import WIDTH
from kev.model.image import ImageEncoder, PatchStem, absorption

torch.manual_seed(0)


def batch() -> dict:
    views = [[torch.randint(0, 256, (3, 40, 70), dtype=torch.uint8)],
             [torch.randint(0, 256, (3, 33, 20), dtype=torch.uint8), torch.randint(0, 256, (3, 16, 16), dtype=torch.uint8)]]
    return collate([{"views": v, "items": []} for v in views])


def encode(model: ImageEncoder, b: dict) -> torch.Tensor:
    return model(b["patches"], b["row"], b["col"], b["view"], b["valid"])


def test_empty_space_and_padding_absorb_nothing():
    assert absorption(torch.tensor([255, 0])).tolist() == [0.0, 1.0]


def test_stem_is_two_convolutions_that_dont_overlap():
    stem = PatchStem()
    patches = torch.randint(0, 256, (5, 3 * 16 * 16), dtype=torch.uint8)
    pixels = absorption(patches).reshape(5, 3, 16, 16)
    tile = stem.tile.weight.reshape(96, 3, 4, 4)
    grid = stem.grid.weight.reshape(WIDTH, 4, 4, 96).permute(0, 3, 1, 2)
    expected = F.conv2d(F.gelu(F.conv2d(pixels, tile, stem.tile.bias, stride=4)), grid, stem.grid.bias, stride=4)
    assert torch.allclose(stem(patches), expected.flatten(1), atol=1e-5)


def test_one_vector_per_patch_and_padding_changes_nothing():
    model, b = ImageEncoder(layers=2), batch()
    out = encode(model, b)
    assert out.shape == (2, 64, WIDTH)
    alone = model(b["patches"][1:, :7], b["row"][1:, :7], b["col"][1:, :7], b["view"][1:, :7], b["valid"][1:, :7])
    assert torch.allclose(out[1, :7], alone[0], atol=1e-5)


def test_listing_order_doesnt_matter_but_position_does():
    model, b = ImageEncoder(layers=2), batch()
    out = encode(model, b)
    shuffle = torch.randperm(64)
    moved = {k: b[k][:1, shuffle] for k in ("patches", "row", "col", "view", "valid")}
    assert torch.allclose(encode(model, moved)[0], out[0, shuffle], atol=1e-5)
    b["row"][0, 0] = 7
    assert not torch.allclose(encode(model, b)[0, 0], out[0, 0], atol=1e-3)


def test_size():
    assert 16.5e6 < sum(p.numel() for p in ImageEncoder().parameters()) < 17e6


def test_checkpointing_gives_the_same_result_and_gradients():
    b = batch()
    model = ImageEncoder(layers=2)
    grads = []
    for checkpointing in (False, True):
        model.zero_grad()
        model.checkpointing = checkpointing
        out = encode(model, b)
        out.sum().backward()
        grads.append((out.detach(), model.stem.tile.weight.grad.clone()))
    assert torch.allclose(grads[0][0], grads[1][0], atol=1e-5) and torch.allclose(grads[0][1], grads[1][1], atol=1e-4)
