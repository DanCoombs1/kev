"""A toy byte-level BPE tokenizer, written to be read rather than to be fast.

BPE (byte-pair encoding) builds a vocabulary by counting. Start with the 256
possible byte values as the first 256 tokens. Find the pair of adjacent tokens
that occurs most often in the training text, add it to the vocabulary as a new
token, and replace it everywhere. Repeat until the vocabulary is the size you
want. Encoding new text replays the same merges in the same order, and decoding
glues the bytes back together, so nothing is ever lost.

This version recounts every pair after every merge, which is slow but easy to
follow. It stays as the reference that the fast version gets checked against.

Usage:
    uv run python src/kev/tokenizer/toy_bpe.py some_text.txt --vocab-size 512
"""

import argparse
import re
from collections import Counter

# Pre-tokenization: before BPE runs, the text is cut into chunks, and a merge can
# never cross a chunk boundary. This is where our own rules live:
#   - a word keeps the single space in front of it, so " the" is one chunk
#   - camelCase and snake_case split into parts: "orderStatus" -> "order", "Status"
#   - every digit is a chunk of its own, so numbers are always spelled digit by digit
# Only ASCII letters count as letters here; anything else falls into the symbols case.
PATTERN = re.compile(
    r" ?[A-Z]?[a-z]+"  # a word, optionally capitalised
    r"| ?[A-Z]+(?![a-z])"  # a run of capitals, like "HTTP" in "HTTPServer"
    r"|\d"  # one digit
    r"| ?[^\sA-Za-z\d]+"  # symbols, punctuation, underscores, non-ASCII
    r"|\s+(?!\S)|\s+"  # whitespace, leaving the last space for the word after it
)

CHUNK_DEMO = 'Call getHTTPServer() for order_4729, "shipped".'
EXAMPLES = [" the Queen said", "orderStatus", "order_status", "4729", " refund", "🐍"]


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace each occurrence of pair in ids with new_id, left to right."""
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
    """Printable form of a token. Bytes that aren't valid UTF-8 on their own show as hex."""
    try:
        return repr(token.decode("utf-8"))
    except UnicodeDecodeError:
        return "<" + token.hex(" ") + ">"


def train(text: str, vocab_size: int) -> tuple[dict[tuple[int, int], int], dict[int, bytes]]:
    """Learn merges from text. Returns the merges in the order learned, and the vocabulary."""
    # Identical chunks always merge identically, so work on each distinct chunk
    # once and weight it by how often it appears.
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
            break  # every chunk is already a single token
        pair = max(pair_counts, key=pair_counts.get)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        chunks = [(merge(ids, pair, new_id), count) for ids, count in chunks]
        left, right, joined = show(vocab[pair[0]]), show(vocab[pair[1]]), show(vocab[new_id])
        print(f"{new_id - 255:5}  {left:>10} + {right:<10} -> {joined:<14} seen {pair_counts[pair]:,}x")
    return merges, vocab


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """Turn text into token ids by replaying the learned merges."""
    ids = []
    for chunk in PATTERN.findall(text):
        chunk_ids = list(chunk.encode("utf-8"))
        # Merges must happen in the order they were learned. Earlier merges got
        # lower ids, so always apply the lowest-id merge available in this chunk.
        while len(chunk_ids) > 1:
            pair = min(zip(chunk_ids, chunk_ids[1:]), key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            chunk_ids = merge(chunk_ids, pair, merges[pair])
        ids.extend(chunk_ids)
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """Turn token ids back into text."""
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a toy BPE tokenizer and show what it learned.")
    parser.add_argument("path", help="a plain-text file to train on")
    parser.add_argument("--vocab-size", type=int, default=512, help="256 byte tokens plus one per merge (default: 512)")
    args = parser.parse_args()

    with open(args.path, encoding="utf-8") as f:
        text = f.read()

    print(f"Pre-tokenization cuts {CHUNK_DEMO!r} into:\n  {PATTERN.findall(CHUNK_DEMO)}\n")
    print("Merges, in the order learned:")
    merges, vocab = train(text, args.vocab_size)

    ids = encode(text, merges)
    assert decode(ids, vocab) == text, "round trip failed"
    n_bytes = len(text.encode("utf-8"))
    print(f"\nTraining text: {n_bytes:,} bytes -> {len(ids):,} tokens ({n_bytes / len(ids):.2f} bytes per token), round trip exact")

    print("\nEncoding examples:")
    for example in EXAMPLES:
        pieces = " ".join(show(vocab[i]) for i in encode(example, merges))
        print(f"  {example!r:>18} -> {pieces}")


if __name__ == "__main__":
    main()
