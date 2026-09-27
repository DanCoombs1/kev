import torch

from kev.data.loader import collate
from kev.model.image import ImageEncoder
from kev.model.kev import ANY_OF, ONE_OF, YES_NO, Kev
from kev.model.text import TextEncoder
from kev.tokenizer.bpe import BYTES, PAD

torch.manual_seed(0)
VOCAB = BYTES + 256 + 50


def model() -> Kev:
    return Kev(ImageEncoder(layers=1), TextEncoder(VOCAB, layers=1), layers=1)


def scan() -> dict:
    views = [[torch.randint(0, 256, (3, 48, 64), dtype=torch.uint8)], [torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)]]
    return collate([{"views": v, "items": []} for v in views])


def words(n: int) -> torch.Tensor:
    return torch.randint(BYTES, VOCAB, (n,))


def questions() -> dict:
    """Scan 0: a yes/no question and a one-of with a "none" option. Scan 1: an any-of, plus a padding question."""
    batch, count, options, tokens = 2, 2, 3, 6
    q = {"ids": torch.full((batch, count, tokens), PAD), "type": torch.tensor([[YES_NO, ONE_OF], [ANY_OF, YES_NO]]),
         "valid": torch.tensor([[True, True], [True, False]]),
         "option_ids": torch.full((batch, count, options, tokens), PAD),
         "is_none": torch.zeros(batch, count, options, dtype=torch.bool),
         "option_valid": torch.tensor([[[True, True, False], [True, True, True]], [[True, True, True], [False] * 3]])}
    for b, n, length in [(0, 0, 5), (0, 1, 6), (1, 0, 4)]:
        q["ids"][b, n, :length] = words(length)
    for b, n, o in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 0, 2)]:
        q["option_ids"][b, n, o, :2] = words(2)
    q["is_none"][0, 1, 2] = True
    q["mask"], q["option_mask"] = q["ids"] != PAD, q["option_ids"] != PAD
    return q


def test_probabilities_follow_the_question_type():
    kev, q = model(), questions()
    scores = kev(scan(), q)
    assert scores.shape == (2, 2, 3) and torch.isfinite(scores).all()
    p = kev.probabilities(scores, q)
    assert torch.allclose(p[0, 0].sum(), torch.tensor(1.0)) and p[0, 0, 2] == 0  # yes/no over its two options
    assert torch.allclose(p[0, 1].sum(), torch.tensor(1.0))  # one of two options, or none
    assert ((p[1, 0] > 0) & (p[1, 0] < 1)).all()  # any-of: each option on its own


def test_the_text_encoder_stays_frozen():
    kev, q = model(), questions()
    before = [p.clone() for p in kev.text.parameters()]
    optimizer = torch.optim.AdamW([p for p in kev.parameters() if p.requires_grad], lr=1e-2)
    kev.train()
    kev(scan(), q).sum().backward()
    optimizer.step()
    assert not kev.text.training and all(p.grad is None for p in kev.text.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, kev.text.parameters()))
    assert kev.question_reader.query.grad is not None and kev.none.grad is not None


def test_none_is_its_own_learned_vector():
    kev, q, s = model(), questions(), scan()
    first = kev(s, q)[0, 1, 2]
    q["option_ids"][0, 1, 2, :3] = words(3)  # words on a "none" option are ignored
    q["option_mask"] = q["option_ids"] != PAD
    assert torch.allclose(kev(s, q)[0, 1, 2], first, atol=1e-5)


def test_padding_questions_change_nothing():
    kev, q, s = model(), questions(), scan()
    kev.eval()
    full = kev(s, q)
    q["ids"][1, 1, :3] = words(3)  # junk in scan 1's padding question
    q["mask"] = q["ids"] != PAD
    assert torch.allclose(kev(s, q)[1, 0], full[1, 0], atol=1e-5)
