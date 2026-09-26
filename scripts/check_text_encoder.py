"""Can the text encoder memorise a handful of sentences? If it can't, something is broken.

    uv run python scripts/check_text_encoder.py
"""

import math
import random
import time

import torch
import torch.nn.functional as F

from kev.device import pick_device
from kev.model.text import TextEncoder
from kev.pretrain.text import mask_words
from kev.questions.bank import training_text
from kev.tokenizer.bpe import MASK, PAD, Tokenizer, pad

STEPS = 300


def fill_in(model: TextEncoder, tokenizer: Tokenizer, words: list[list[int]], w: int, device: torch.device) -> str:
    ids = [i for word in words[:w] for i in word] + [MASK] * len(words[w]) + [i for word in words[w + 1:] for i in word]
    at = sum(map(len, words[:w]))
    with torch.no_grad():
        x = model(torch.tensor([ids], device=device), torch.ones(1, len(ids), dtype=torch.bool, device=device))
        guess = model.predict(x[0, at: at + len(words[w])]).argmax(dim=-1)
    return tokenizer.decode(guess.tolist())


def score(model, tokenizer, sentences, device) -> tuple[int, int]:
    right = total = 0
    for s in sentences:
        words = tokenizer.encode_words(s)
        for w in range(len(words)):
            right += fill_in(model, tokenizer, words, w, device) == tokenizer.decode(words[w])
            total += 1
    return right, total


def main() -> None:
    device = pick_device()
    rng = random.Random(0)
    torch.manual_seed(0)
    tokenizer = Tokenizer.load()
    sentences = rng.sample([t for t in training_text() if len(t.split()) >= 5], 10)
    model = TextEncoder(len(tokenizer)).to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, vocabulary {len(tokenizer):,}")
    print(f"a blind guess over the vocabulary would score a loss of ln({len(tokenizer):,}) = {math.log(len(tokenizer)):.2f}\n")

    right, total = score(model, tokenizer, sentences, device)
    print(f"before training: fills in {right} of {total} words\n")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    start = time.perf_counter()
    for step in range(STEPS + 1):
        inputs, targets = zip(*(mask_words(tokenizer.encode_words(s), rng, len(tokenizer)) for s in sentences))
        inputs, targets = pad(list(inputs))[0].to(device), pad(list(targets))[0].to(device)
        hidden = targets != PAD
        loss = F.cross_entropy(model.predict(model(inputs, inputs != PAD)[hidden]), targets[hidden])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"step {step:3d}  loss {loss.item():.3f}")
    print(f"({time.perf_counter() - start:.0f}s)\n")

    right, total = score(model, tokenizer, sentences, device)
    print(f"after training: fills in {right} of {total} words\n")
    for s in sentences[:5]:
        words = tokenizer.encode_words(s)
        w = rng.randrange(len(words))
        blank = " [___]" if tokenizer.decode(words[w]).startswith(" ") else "[___]"
        shown = "".join(blank if i == w else tokenizer.decode(word) for i, word in enumerate(words))
        print(f"  {shown.strip()!r:58} -> {fill_in(model, tokenizer, words, w, device)!r}")


if __name__ == "__main__":
    main()
