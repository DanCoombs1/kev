"""Byte-level BPE, kept simple as a reference implementation.

    uv run python src/kev/tokenizer/toy_bpe.py corpus.txt --vocab-size 512
"""

import argparse
import re
from collections import Counter

# Merges never cross chunk boundaries. Words keep their leading space, camelCase and
# snake_case are split into parts, and every digit is its own chunk.
PATTERN = re.compile(
    r" ?[A-Z]?[a-z]+"
    r"| ?[A-Z]+(?![a-z])"
    r"|\d"
    r"| ?[^\sA-Za-z\d]+"
    r"|\s+(?!\S)|\s+"
)

EXAMPLES = ["orderStatus", "order_status", "4729", " refund", "🐍"]


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def show(token: bytes) -> str:
    try:
        return repr(token.decode("utf-8"))
    except UnicodeDecodeError:
        return "<" + token.hex(" ") + ">"


def train(text: str, vocab_size: int) -> tuple[dict[tuple[int, int], int], dict[int, bytes]]:
    chunk_counts = Counter(PATTERN.findall(text))
    chunks = [(list(chunk.encode("utf-8")), count) for chunk, count in chunk_counts.items()]

    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}
    for new_id in range(256, vocab_size):
        pair_counts = Counter()
        for ids, count in chunks:
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += count
        if not pair_counts:
            break
        pair = max(pair_counts, key=pair_counts.get)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        chunks = [(merge(ids, pair, new_id), count) for ids, count in chunks]
        left, right, joined = show(vocab[pair[0]]), show(vocab[pair[1]]), show(vocab[new_id])
        print(f"{new_id - 255:5}  {left:>10} + {right:<10} -> {joined:<14} {pair_counts[pair]:,}")
    return merges, vocab


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    ids = []
    for chunk in PATTERN.findall(text):
        chunk_ids = list(chunk.encode("utf-8"))
        # Lower ids were learned earlier, so this replays merges in training order.
        while len(chunk_ids) > 1:
            pair = min(zip(chunk_ids, chunk_ids[1:]), key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            chunk_ids = merge(chunk_ids, pair, merges[pair])
        ids.extend(chunk_ids)
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--vocab-size", type=int, default=512)
    args = parser.parse_args()

    with open(args.path, encoding="utf-8") as f:
        text = f.read()

    merges, vocab = train(text, args.vocab_size)
    ids = encode(text, merges)
    assert decode(ids, vocab) == text
    n_bytes = len(text.encode("utf-8"))
    print(f"\n{n_bytes:,} bytes -> {len(ids):,} tokens ({n_bytes / len(ids):.2f} bytes/token)\n")
    for example in EXAMPLES:
        print(f"{example!r:>18} -> {' '.join(show(vocab[i]) for i in encode(example, merges))}")


if __name__ == "__main__":
    main()
