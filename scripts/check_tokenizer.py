"""Tokens per word at different vocabulary sizes, what the merges look like, and how questions get split.

    uv run python scripts/check_tokenizer.py
"""

from kev.data.text import squad_questions, wikitext
from kev.questions.bank import held_out_questions, training_text
from kev.tokenizer.bpe import BYTES, Tokenizer

SIZES = [1000, 2000, 4000, 8000, 16000]
WIDTH = 384
EXAMPLES = [
    "Is there a knife in this bag?",
    "check for 3D-printed pistols",
    "any modified walkie-talkies?",
    "is someone carrying a powerbank?",
    "do you reckon there are spanners in here?",
    "does this luggage have a beeper inside?",
]


def tokens_per_word(tokenizer: Tokenizer, texts: list[str]) -> float:
    return sum(len(tokenizer.encode(t)) for t in texts) / sum(len(t.split()) for t in texts)


def show(tokenizer: Tokenizer, token: int) -> str:
    return repr(tokenizer.vocab[token].decode("utf-8", errors="replace"))


def main() -> None:
    texts = {
        "wikipedia": list(wikitext("validation")),
        "squad": squad_questions("validation"),
        "bank": training_text(),
        "held out": [q for qs in held_out_questions().values() for q in qs],
    }
    full = Tokenizer.load(merges=None)
    print(f"{'merges':>7} {'params':>7}  " + "  ".join(f"{name:>9}" for name in texts))
    for n in SIZES:
        tokenizer = Tokenizer.load(merges=n)
        rates = "  ".join(f"{tokens_per_word(tokenizer, t):9.2f}" for t in texts.values())
        print(f"{n:7,} {len(tokenizer) * WIDTH / 1e6:6.1f}M  {rates}")

    for start in [0, 1000, 2000, 4000, 8000, 15990]:
        merges = range(start, start + 10)
        print(f"\nmerges {start:,}-{start + 9:,}: " + "  ".join(show(full, BYTES + 256 + i) for i in merges))

    for n in [1000, 4000, 8000, 16000]:
        tokenizer = Tokenizer.load(merges=n)
        print(f"\n{n:,} merges")
        for text in EXAMPLES:
            print(f"  {' | '.join(tokenizer.pieces(text))}")


if __name__ == "__main__":
    main()
