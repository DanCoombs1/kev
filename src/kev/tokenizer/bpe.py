"""Byte-level BPE for kev's questions and options.

Text is normalised (NFKC, lowercase, single spaces) and split into chunks that merges never cross. Each chunk's
UTF-8 bytes are then merged in the order the merges were learned. Special tokens are only added by question(),
option(), none_option() and the text pretraining, never read from text, so typing "[q]" can't produce the question
marker.

    uv run python -m kev.tokenizer.bpe      (trains on data/text and the question bank, writes data/tokenizer.json)
"""

import heapq
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import batched
from multiprocessing import Pool
from pathlib import Path

import torch

from kev.data.sources import DATA

SPECIALS = ["[pad]", "[q]", "[o]", "[none]", "[mask]"]
PAD, QUESTION, OPTION, NONE, MASK = range(len(SPECIALS))
BYTES = len(SPECIALS)  # id of byte 0; merges start at BYTES + 256

# Words keep their leading space, contractions stay together, and every digit is its own chunk.
PATTERN = re.compile(r"'(?:s|t|re|ve|m|ll|d)\b| ?[^\W\d_]+| ?\d| ?[^\s\w]+|_+|\s+(?!\S)|\s+")

TOKENIZER = DATA / "tokenizer.json"
MAX_MERGES = 16000
MERGES = 8000  # the vocabulary kev uses, chosen with scripts/check_tokenizer.py
SQUAD_WEIGHT = 5
BANK_WEIGHT = 100


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = merges
        self.rank = {pair: BYTES + 256 + i for i, pair in enumerate(merges)}
        self.vocab = [s.encode() for s in SPECIALS] + [bytes([b]) for b in range(256)]
        for a, b in merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.cache = {}

    def __len__(self) -> int:
        return len(self.vocab)

    def _chunk(self, chunk: str) -> list[int]:
        ids = self.cache.get(chunk)
        if ids is None:
            ids = [BYTES + b for b in chunk.encode()]
            while len(ids) > 1:
                # The earliest-learned merge goes first, as in training.
                pair = min(zip(ids, ids[1:]), key=lambda p: self.rank.get(p, float("inf")))
                if pair not in self.rank:
                    break
                ids = merge(ids, pair, self.rank[pair])
            self.cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        return [i for chunk in PATTERN.findall(normalize(text)) for i in self._chunk(chunk)]

    def encode_words(self, text: str) -> list[list[int]]:
        """Token ids grouped by chunk, so a word's pieces can be hidden together."""
        return [self._chunk(chunk) for chunk in PATTERN.findall(normalize(text))]

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[i] for i in ids if i >= BYTES).decode("utf-8", errors="replace")

    def pieces(self, text: str) -> list[str]:
        return [self.vocab[i].decode("utf-8", errors="replace") for i in self.encode(text)]

    def question(self, text: str) -> list[int]:
        return [QUESTION] + self.encode(text)

    def option(self, text: str) -> list[int]:
        return [OPTION] + self.encode(text)

    def none_option(self) -> list[int]:
        return [OPTION, NONE]

    def save(self, path: Path = TOKENIZER) -> None:
        path.write_text(json.dumps({"specials": SPECIALS, "merges": self.merges}))

    @classmethod
    def load(cls, path: Path = TOKENIZER, merges: int | None = MERGES) -> "Tokenizer":
        data = json.loads(path.read_text())
        if data["specials"] != SPECIALS:
            raise ValueError(f"{path} was saved with different special tokens")
        # The first n merges of a longer run are exactly what a run stopped at n would have learned.
        return cls([tuple(pair) for pair in data["merges"][:merges]])


def pad(sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.full((len(sequences), max(map(len, sequences))), PAD, dtype=torch.long)
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(seq)
    return ids, ids != PAD


def count_chunks(texts: Iterable[str]) -> Counter:
    counts = Counter()
    for text in texts:
        counts.update(PATTERN.findall(normalize(text)))
    return counts


def train(chunk_counts: Counter, n_merges: int, report: int = 0) -> list[tuple[int, int]]:
    """Keeps a running count of every pair and, after a merge, only revisits the chunks that contained it."""
    words = [[BYTES + b for b in chunk.encode()] for chunk in chunk_counts]
    weights = list(chunk_counts.values())
    pairs = Counter()
    where = defaultdict(set)  # pair -> chunks containing it
    for w, (ids, n) in enumerate(zip(words, weights)):
        for pair in zip(ids, ids[1:]):
            pairs[pair] += n
            where[pair].add(w)
    # Largest count first, ties to the smallest pair. Entries go stale when counts change; those get skipped.
    heap = [(-n, pair) for pair, n in pairs.items()]
    heapq.heapify(heap)

    merges = []
    start = time.perf_counter()
    while heap and len(merges) < n_merges:
        n, pair = heapq.heappop(heap)
        if -n != pairs.get(pair, 0):
            continue
        new_id = BYTES + 256 + len(merges)
        merges.append(pair)
        changed = set()
        for w in where.pop(pair):
            ids, weight = words[w], weights[w]
            for p in zip(ids, ids[1:]):
                pairs[p] -= weight
                where[p].discard(w)
                changed.add(p)
            words[w] = ids = merge(ids, pair, new_id)
            for p in zip(ids, ids[1:]):
                pairs[p] += weight
                where[p].add(w)
                changed.add(p)
        for p in changed:
            if pairs[p] > 0:
                heapq.heappush(heap, (-pairs[p], p))
            else:
                del pairs[p]
                where.pop(p, None)
        if report and len(merges) % report == 0:
            print(f"  {len(merges):,} merges, {time.perf_counter() - start:.0f}s")
    return merges


def main() -> None:
    from kev.data.text import squad_questions, wikitext
    from kev.questions.bank import training_text

    start = time.perf_counter()
    counts = Counter()
    with Pool() as pool:
        for part in pool.imap_unordered(count_chunks, batched(wikitext("train"), 10_000)):
            counts.update(part)
    for chunk, n in count_chunks(squad_questions("train")).items():
        counts[chunk] += n * SQUAD_WEIGHT
    for chunk, n in count_chunks(training_text()).items():
        counts[chunk] += n * BANK_WEIGHT
    print(f"{len(counts):,} distinct chunks, {sum(counts.values()) / 1e6:.0f}M in all ({time.perf_counter() - start:.0f}s)")

    tokenizer = Tokenizer(train(counts, MAX_MERGES, report=2000))
    tokenizer.save()
    print(f"{len(tokenizer):,} tokens -> {TOKENIZER.name} ({time.perf_counter() - start:.0f}s)")


if __name__ == "__main__":
    main()
