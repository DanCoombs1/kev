"""English text for the tokenizer and text pretraining: WikiText-103 articles and SQuAD questions.

WikiText comes pre-split, with spaces around punctuation ("red @-@ tipped , while"). detokenize() puts it back
into normal text, like the questions kev will be asked.
"""

import re
from collections.abc import Iterator

import pyarrow.parquet as pq

from kev.data.sources import DATA

TEXT = DATA / "text"


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


def squad_questions(split: str) -> list[str]:
    return list(dict.fromkeys(q.strip() for q in _column(f"squad/plain_text/{split}-*.parquet", "question")))
