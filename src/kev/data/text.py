"""English text for the tokenizer and text pretraining: WikiText-103 articles, SQuAD questions, Simple English
Wikipedia, and paragraphs of the full English Wikipedia about the kinds of things kev is asked about.

WikiText comes pre-split, with spaces around punctuation ("red @-@ tipped , while"). detokenize() puts it back
into normal text, like the questions kev will be asked.

    uv run python -m kev.data.text     (picks the full-Wikipedia paragraphs, writes data/text/wikipedia_topics.parquet)
"""

import random
import re
import time
from collections import Counter
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from kev.data.sources import DATA

TEXT = DATA / "text"
WIKIPEDIA = TEXT / "wikipedia"
TOPICS = TEXT / "wikipedia_topics.parquet"
TOPIC_CAP = 10_000  # paragraphs per keyword, so "security" and "airport" don't crowd out "pliers"
MIN_WORDS = 20

# Paragraphs are picked by the words of kev's training wording plus these, never by held-out wording, so the
# held-out test stays fair.
TOPIC_WORDS = ["tool", "tools", "weapon", "weapons", "airport", "airports", "baggage", "security", "passenger",
               "passengers", "x-ray", "screening", "device", "devices", "electronics", "hardware", "household",
               "kitchen", "gadget", "gadgets"]


def detokenize(line: str) -> str:
    line = re.sub(r" @(.)@ ", r"\1", line)
    line = re.sub(r'" (.*?) "', r'"\1"', line)
    line = re.sub(r" ([.,;:?!)\]%]|'\w)", r"\1", line)
    line = re.sub(r"([(\[]) ", r"\1", line)
    return line.strip()


def _column(pattern: str, column: str) -> Iterator[str]:
    for path in sorted(TEXT.glob(pattern)):
        for batch in pq.ParquetFile(path).iter_batches(columns=[column]):
            yield from batch.column(0).to_pylist()


def wikitext(split: str) -> Iterator[str]:
    """Paragraphs, without the headings."""
    for line in _column(f"wikitext/wikitext-103-raw-v1/{split}-*.parquet", "text"):
        line = line.strip()
        if line and not line.startswith("= "):
            yield detokenize(line)


def wikitext_titles(splits: tuple[str, ...]) -> set[str]:
    titles = set()
    for split in splits:
        for line in _column(f"wikitext/wikitext-103-raw-v1/{split}-*.parquet", "text"):
            line = line.strip()
            if line.startswith("= ") and not line.startswith("= ="):
                titles.add(detokenize(line.strip("= ")).lower())
    return titles


def squad_questions(split: str) -> list[str]:
    return list(dict.fromkeys(q.strip() for q in _column(f"squad/plain_text/{split}-*.parquet", "question")))


def _articles(path: Path) -> Iterator[tuple[str, str]]:
    for batch in pq.ParquetFile(path).iter_batches(columns=["title", "text"]):
        yield from zip(batch.column(0).to_pylist(), batch.column(1).to_pylist())


def simple_wikipedia() -> Iterator[str]:
    # Leaving out the articles WikiText checks against, so their content can't help the validation score.
    skip = wikitext_titles(("validation", "test"))
    for path in sorted((WIKIPEDIA / "20231101.simple").glob("*.parquet")):
        for title, text in _articles(path):
            if title.lower() not in skip:
                yield from (p.strip() for p in text.split("\n") if len(p.split()) >= 8)


def topic_words() -> set[str]:
    from kev.questions.bank import GROUP_NOUNS, ITEMS

    nouns = [n for table in (ITEMS, GROUP_NOUNS) for group in table.values() for n in group]
    return {w.lower() for n in nouns for w in (n.word, n.many)} | set(TOPIC_WORDS)


def _candidates(job: tuple[Path, set[str], set[str]]) -> list[tuple[str, list[str]]]:
    path, words, skip = job
    pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, words), key=len, reverse=True)) + r")\b")
    found = []
    for title, text in _articles(path):
        if title.lower() in skip:
            continue
        for paragraph in text.split("\n"):
            if len(paragraph.split()) >= MIN_WORDS:
                hits = set(pattern.findall(paragraph.lower()))
                if hits:
                    found.append((paragraph.strip(), sorted(hits)))
    return found


def build_topics(seed: int = 0) -> None:
    """Keeps paragraphs that mention a topic word, at most about TOPIC_CAP per word, counting each paragraph under
    its rarest word. All of WikiText's articles are left out: they're already in the training text or checked against."""
    words, skip = topic_words(), wikitext_titles(("train", "validation", "test"))
    paths = sorted((WIKIPEDIA / "20231101.en").glob("*.parquet"))
    with Pool(6) as pool:  # each worker holds a whole file, so not one per core
        files = pool.map(_candidates, [(p, words, skip) for p in paths])
    counts = Counter(w for file in files for _, hits in file for w in hits)
    rng = random.Random(seed)
    kept, per_word = [], Counter()
    for file in files:
        for paragraph, hits in file:
            rarest = min(hits, key=lambda w: (counts[w], w))
            if rng.random() < TOPIC_CAP / counts[rarest]:
                kept.append(paragraph)
                per_word[rarest] += 1
    pq.write_table(pa.table({"text": kept}), TOPICS)
    print(f"{len(paths)} files: kept {len(kept):,} of {sum(map(len, files)):,} paragraphs, "
          f"{sum(len(p.split()) for p in kept) / 1e6:.0f}M words")
    print("most kept:", ", ".join(f"{w} {n:,}" for w, n in per_word.most_common(8)))
    print("rarest:", ", ".join(f"{w} {per_word[w]:,}" for w in sorted(per_word, key=per_word.get)[:8]))


def wikipedia_topics() -> list[str]:
    return pq.read_table(TOPICS).column("text").to_pylist()


if __name__ == "__main__":
    start = time.perf_counter()
    build_topics()
    print(f"({time.perf_counter() - start:.0f}s)")
