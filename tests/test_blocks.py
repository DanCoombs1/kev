import math

import torch

from kev.model.blocks import HEADS, WIDTH, Attention, Block

torch.manual_seed(0)


def by_hand(attention: Attention, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    query = attention.query(x)
    key, value = attention.key_value(x).chunk(2, dim=-1)
    head = WIDTH // HEADS
    heads = []
    for h in range(HEADS):
        part = slice(h * head, (h + 1) * head)
        scores = query[..., part] @ key[..., part].transpose(1, 2) / math.sqrt(head)  # (batch, tokens, tokens)
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        heads.append(scores.softmax(dim=-1) @ value[..., part])
    return attention.out(torch.cat(heads, dim=-1))


def test_attention_is_the_softmax_formula():
    attention = Attention()
    x = torch.randn(2, 7, WIDTH)
    mask = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])
    assert torch.allclose(attention(x, mask), by_hand(attention, x, mask), atol=1e-5)


def test_block_keeps_the_shape_and_has_the_expected_size():
    block = Block()
    assert block(torch.randn(3, 11, WIDTH)).shape == (3, 11, WIDTH)
    assert sum(p.numel() for p in block.parameters()) == 1_774_464


def test_padding_cannot_change_the_real_tokens():
    block = Block()
    real = torch.randn(1, 5, WIDTH)
    padded = torch.cat([real, 100 * torch.randn(1, 3, WIDTH)], dim=1)
    mask = torch.tensor([[True] * 5 + [False] * 3])
    assert torch.allclose(block(padded, mask)[:, :5], block(real), atol=1e-5)


def test_block_alone_cannot_see_word_order():
    block = Block()
    x = torch.randn(1, 6, WIDTH)
    shuffle = torch.randperm(6)
    assert torch.allclose(block(x[:, shuffle]), block(x)[:, shuffle], atol=1e-5)


def test_cross_attention_reads_the_context():
    attention = Attention()
    questions, patches = torch.randn(2, 4, WIDTH), torch.randn(2, 9, WIDTH)
    mask = torch.tensor([[True] * 9, [True] * 6 + [False] * 3])
    out = attention(questions, mask, context=patches)
    assert out.shape == (2, 4, WIDTH)
    assert torch.allclose(out[1], attention(questions[1:], mask[1:, :6], context=patches[1:, :6])[0], atol=1e-5)
