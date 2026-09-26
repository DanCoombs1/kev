"""Pretrains the text encoder by hiding words and having it fill them in."""

import random

from kev.tokenizer.bpe import BYTES, MASK, PAD

MASK_RATE = 0.15


def mask_words(words: list[list[int]], rng: random.Random, vocab: int) -> tuple[list[int], list[int]]:
    """Hides about 15% of the words, all pieces of a word together. Of those, 80% become [mask], 10% a random
    token and 10% stay as they are, since real questions never contain [mask]. Targets are PAD where nothing
    is hidden."""
    hidden = set(rng.sample(range(len(words)), max(1, round(MASK_RATE * len(words)))))
    inputs, targets = [], []
    for w, word in enumerate(words):
        if w not in hidden:
            inputs += word
            targets += [PAD] * len(word)
            continue
        roll = rng.random()
        for token in word:
            inputs.append(MASK if roll < 0.8 else rng.randrange(BYTES, vocab) if roll < 0.9 else token)
            targets.append(token)
    return inputs, targets
