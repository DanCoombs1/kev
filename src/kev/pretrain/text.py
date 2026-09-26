"""Pretrains the text encoder by hiding words and having it fill them in.

    uv run python -m kev.pretrain.text     (encodes data/text once, then trains; writes runs/text/)
"""

import json
import math
import random
import time
from itertools import batched
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kev.data.loader import BucketSampler
from kev.data.sources import DATA
from kev.device import pick_device
from kev.model.text import MAX_TOKENS, TextEncoder
from kev.tokenizer.bpe import BYTES, MASK, MERGES, PAD, Tokenizer, pad

MASK_RATE = 0.15
TOKENS = DATA / "text_tokens"
RUN = DATA.parent / "runs" / "text"
SQUAD_REPEATS = 3
BANK_REPEATS = 10
BATCH = 128
PASSES = 1
PEAK_LR = 5e-4
WARMUP = 0.02
WEIGHT_DECAY = 0.01
WORKERS = 6
VALIDATE_EVERY = 2000
VALIDATION_SEQUENCES = 2000


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


_tokenizer = None


def _encode(texts: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Token ids, word starts, and sequence lengths, cutting texts into sequences at word boundaries."""
    global _tokenizer
    _tokenizer = _tokenizer or Tokenizer.load()
    tokens, starts, lengths = [], [], []
    for text in texts:
        length = 0
        for word in _tokenizer.encode_words(text):
            word = word[:MAX_TOKENS]
            if length + len(word) > MAX_TOKENS:
                lengths.append(length)
                length = 0
            tokens += word
            starts += [True] + [False] * (len(word) - 1)
            length += len(word)
        if length:
            lengths.append(length)
    return np.array(tokens, dtype=np.int16), np.array(starts, dtype=bool), lengths


def encode_texts(texts: list[str], out: Path) -> None:
    with Pool() as pool:
        parts = pool.map(_encode, batched(texts, 2000))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "tokens.npy", np.concatenate([p[0] for p in parts]))
    np.save(out / "starts.npy", np.concatenate([p[1] for p in parts]))
    lengths = [n for p in parts for n in p[2]]
    np.save(out / "offsets.npy", np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))


def corpora() -> dict[str, list[str]]:
    from kev.data.text import simple_wikipedia, squad_questions, wikipedia_topics, wikitext
    from kev.questions.bank import training_text

    bank = [t for t in training_text() if len(t.split()) >= 3]
    return {
        "train": [*wikitext("train"), *simple_wikipedia(), *wikipedia_topics(), *squad_questions("train") * SQUAD_REPEATS,
                  *bank * BANK_REPEATS],
        "val_wikipedia": list(wikitext("validation")),
        "val_questions": squad_questions("validation"),
    }


class Sequences(torch.utils.data.Dataset):
    """Encoded sequences with words hidden afresh each time, or the same way every time when `fixed`."""

    def __init__(self, split: str, vocab: int, fixed: bool = False, limit: int | None = None):
        self.path = TOKENS / split
        self.lengths = np.diff(np.load(self.path / "offsets.npy"))[:limit]
        self.vocab = vocab
        self.fixed = fixed
        self.arrays = None

    def __getstate__(self) -> dict:
        return {**self.__dict__, "arrays": None}  # each worker maps the files itself

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, k: int) -> tuple[list[int], list[int]]:
        if self.arrays is None:
            self.arrays = [np.load(self.path / f"{name}.npy", mmap_mode="r") for name in ("tokens", "starts", "offsets")]
        tokens, starts, offsets = self.arrays
        a, b = offsets[k], offsets[k + 1]
        ids = tokens[a:b].tolist()
        cuts = np.flatnonzero(starts[a:b]).tolist() + [b - a]
        words = [ids[i:j] for i, j in zip(cuts, cuts[1:])]
        return mask_words(words, random.Random(k) if self.fixed else random.Random(), self.vocab)


def collate(samples: list[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, targets = zip(*samples)
    return pad(list(inputs))[0], pad(list(targets))[0]


def learning_rate(step: int, total: int) -> float:
    """A short linear warm-up, then a cosine curve down to a tenth of the peak."""
    warm = max(1, int(WARMUP * total))
    if step < warm:
        return PEAK_LR * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return PEAK_LR * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def masked_loss(model: TextEncoder, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = targets != PAD
    logits = model.predict(model(inputs, inputs != PAD)[hidden])
    right = logits.argmax(dim=-1) == targets[hidden]
    return F.cross_entropy(logits, targets[hidden], reduction="sum"), right.sum()


@torch.no_grad()
def validate(model: TextEncoder, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, float]:
    total_loss = total_right = count = 0
    for inputs, targets in loader:
        loss, right = masked_loss(model, inputs.to(device), targets.to(device))
        total_loss, total_right, count = total_loss + loss.item(), total_right + right.item(), count + (targets != PAD).sum().item()
    return total_loss / count, total_right / count


def train() -> None:
    device = pick_device()
    torch.manual_seed(0)
    vocab = len(Tokenizer.load())
    data = Sequences("train", vocab)
    sampler = BucketSampler(data.lengths, BATCH)
    loader = torch.utils.data.DataLoader(data, batch_sampler=sampler, collate_fn=collate, num_workers=WORKERS,
                                         persistent_workers=True)
    checks = {name: torch.utils.data.DataLoader(Sequences(name, vocab, fixed=True, limit=VALIDATION_SEQUENCES),
                                                batch_size=BATCH, collate_fn=collate)
              for name in ("val_wikipedia", "val_questions")}

    model = TextEncoder(vocab).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
                                  lr=PEAK_LR, betas=(0.9, 0.98))
    total = PASSES * len(sampler)
    print(f"{len(data):,} sequences, {int(data.lengths.sum()):,} tokens, {total:,} steps")

    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / "log.jsonl", "w") as log:
        def record(entry: dict) -> None:
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print("  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in entry.items()), flush=True)

        step, seen, start = 0, 0, time.perf_counter()
        running_loss = running_right = running_count = 0
        for _ in range(PASSES):
            for inputs, targets in loader:
                lr = learning_rate(step, total)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                count = int((targets != PAD).sum())
                seen += int((inputs != PAD).sum())
                loss, right = masked_loss(model, inputs.to(device), targets.to(device))
                optimizer.zero_grad(set_to_none=True)
                (loss / count).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1
                running_loss, running_right, running_count = running_loss + loss.detach(), running_right + right, running_count + count

                if step % 200 == 0:
                    elapsed = time.perf_counter() - start
                    record({"step": step, "loss": running_loss.item() / running_count, "accuracy": running_right.item() / running_count,
                            "lr": lr, "tokens_per_s": round(seen / elapsed), "minutes": round(elapsed / 60, 1)})
                    running_loss = running_right = running_count = 0
                if step % VALIDATE_EVERY == 0 or step == total:
                    entry = {"step": step}
                    for name, check in checks.items():
                        entry[f"{name}_loss"], entry[f"{name}_accuracy"] = validate(model, check, device)
                    record(entry)
                    torch.save({"model": model.state_dict(), "step": step, "vocab": vocab, "merges": MERGES}, RUN / "encoder.pt")


def main() -> None:
    for split, texts in corpora().items():
        if not (TOKENS / split / "offsets.npy").exists():
            start = time.perf_counter()
            encode_texts(texts, TOKENS / split)
            print(f"{split}: encoded {len(texts):,} texts ({time.perf_counter() - start:.0f}s)")
    train()


if __name__ == "__main__":
    main()
