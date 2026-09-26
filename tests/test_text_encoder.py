import itertools
import random

import torch

from kev.model.blocks import WIDTH
from kev.model.text import MAX_TOKENS, TextEncoder
from kev.pretrain.text import PEAK_LR, _encode, learning_rate, mask_words
from kev.tokenizer.bpe import BYTES, MASK, PAD, Tokenizer

torch.manual_seed(0)
VOCAB = BYTES + 256 + 100


def test_positions_let_it_see_word_order():
    model = TextEncoder(VOCAB)
    ids = torch.randint(BYTES, VOCAB, (1, 6))
    shuffle = torch.tensor([1, 0, 2, 3, 5, 4])
    mask = torch.ones(1, 6, dtype=torch.bool)
    assert not torch.allclose(model(ids[:, shuffle], mask), model(ids, mask)[:, shuffle], atol=1e-3)


def test_padding_cannot_change_the_real_tokens():
    model = TextEncoder(VOCAB)
    ids = torch.randint(BYTES, VOCAB, (1, 5))
    padded = torch.cat([ids, torch.full((1, 3), PAD)], dim=1)
    out = model(padded, padded != PAD)
    assert out.shape == (1, 8, WIDTH)
    assert torch.allclose(out[:, :5], model(ids, ids != PAD), atol=1e-5)
    assert model.predict(out).shape == (1, 8, VOCAB)


def test_whole_words_are_hidden_about_15_percent_of_the_time():
    rng = random.Random(0)
    words, hidden, masked = 0, 0, 0
    for _ in range(2000):
        sentence = [[rng.randrange(BYTES, VOCAB) for _ in range(rng.randint(1, 3))] for _ in range(rng.randint(3, 30))]
        inputs, targets = mask_words(sentence, rng, VOCAB)
        assert len(inputs) == len(targets) == sum(map(len, sentence))
        at = 0
        for word in sentence:
            span = targets[at: at + len(word)]
            assert span == word or span == [PAD] * len(word)  # all of a word or none of it
            if span == word:
                hidden += 1
                masked += inputs[at] == MASK
            at += len(word)
        words += len(sentence)
    assert 0.14 < hidden / words < 0.17
    assert 0.77 < masked / hidden < 0.83


def test_texts_are_cut_at_word_boundaries_into_short_enough_sequences():
    text = " ".join(["the spanner is in the luggage"] * 60)
    tokens, starts, lengths = _encode((text, "is there a knife?"))
    assert max(lengths) <= MAX_TOKENS and sum(lengths) == len(tokens) == len(starts)
    offsets = [0, *itertools.accumulate(lengths)]
    assert all(starts[a] for a in offsets[:-1])  # every sequence begins with a whole word
    assert Tokenizer.load().decode(tokens[offsets[-2]:].tolist()) == "is there a knife?"


def test_learning_rate_warms_up_then_decays_to_a_tenth():
    rates = [learning_rate(step, 1000) for step in range(1000)]
    assert rates[0] < rates[10] < rates[19] == PEAK_LR
    assert all(a >= b for a, b in zip(rates[20:], rates[21:]))
    assert abs(rates[-1] - PEAK_LR / 10) < 1e-6
