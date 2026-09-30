"""Picks the questions a scan is asked, and their answers.

Items are picked without looking at the bag, so "no" is as common as in real bags. Only items the scan's dataset labels
are asked about, since only those have trustworthy answers. Every scan also gets one any-of question listing all of
them, so each item in the bag says "yes" somewhere without the question being chosen for it.

    uv run python -m kev.questions.generate     (prints questions for a few training scans)
"""

import random
from dataclasses import dataclass

import torch

from kev.data.names import NAMES, RESTRICTED, WEAPON
from kev.model.kev import ANY_OF, ONE_OF, YES_NO
from kev.questions.bank import (
    ABSENT, ABSENT_HELD_OUT, GROUP_NOUNS, GROUP_NOUNS_HELD_OUT, ITEMS, ITEMS_HELD_OUT, NO, PRESENT, PRESENT_HELD_OUT,
    WHICH, WHICH_ALL, WHICH_ALL_HELD_OUT, WHICH_HELD_OUT, YES, Noun, ask,
)
from kev.tokenizer.bpe import PAD, Tokenizer

QUESTIONS_PER_SCAN = 8
KINDS = {YES_NO: 0.5, ONE_OF: 0.25, ANY_OF: 0.25}
NEGATED = 0.25  # of yes/no questions worded the other way round ("is this bag free of knives?")
OPTION_COUNTS = {ONE_OF: (2, 4), ANY_OF: (2, 5)}
TRIES = 10

ASKABLE = {}
for (dataset, _), name in NAMES.items():
    ASKABLE.setdefault(dataset, set()).add(name)
ASKABLE = {dataset: sorted(names) for dataset, names in ASKABLE.items()}

# A bag holding the key also holds the value: "is there a gun?" on a 3D-printed gun is yes.
IS_A = {"3D-printed gun": "gun", "modified laptop": "laptop", "modified mobile phone": "mobile phone",
        "modified pager": "pager", "modified walkie-talkie": "walkie-talkie"}

# Either could fairly describe the other, so with one in the bag the other gets neither yes nor no.
MODIFIED = ["modified laptop", "modified mobile phone", "modified pager", "modified walkie-talkie", "modified electronic parts"]
SHARP = ["knife", "box cutter", "razor blade", "razor", "saw blade", "scissors"]
OVERLAPS = {frozenset(pair) for pair in [("knife", "box cutter"), ("razor", "razor blade"), ("battery", "power bank"),
                                         *[("other sharp object", s) for s in SHARP], *[("explosive", m) for m in MODIFIED]]}


@dataclass(frozen=True)
class Wording:
    items: dict[str, list[Noun]]
    groups: dict[str, list[Noun]]
    present: list[str]
    absent: list[str]
    which: list[str]
    which_all: list[str]


TRAINING = Wording(ITEMS, GROUP_NOUNS, PRESENT, ABSENT, WHICH, WHICH_ALL)
HELD_OUT = Wording(ITEMS_HELD_OUT, GROUP_NOUNS_HELD_OUT, PRESENT_HELD_OUT, ABSENT_HELD_OUT, WHICH_HELD_OUT, WHICH_ALL_HELD_OUT)


@dataclass
class Question:
    type: int
    text: str
    options: list[str | None]  # None is "none of these"
    answer: list[bool]  # which options are right; exactly one for yes_no and one_of
    scored: list[bool]  # options whose answer is known; any_of leaves ambiguous ones out
    about: list[str | None]  # the item or group each option is about (yes/no: the one asked about)
    negated: bool = False
    template: str = ""  # the sentence it was made from


def present(sample: dict) -> set[str]:
    names = {i["name"] for i in sample["items"]}
    return names | {IS_A[n] for n in names if n in IS_A}


def unclear(item: str, there: set[str]) -> bool:
    return item not in there and any(frozenset((item, p)) in OVERLAPS for p in there)


def asks_about_items(sample: dict) -> bool:
    if sample["dataset"] not in ASKABLE or sample["yes_only"]:  # yes-only trays: a "no" could be wrong
        return False
    return not (sample["dataset"] == "dvxray" and not sample["items"])  # clean DvXray bags give themselves away


def _noun(table: dict[str, list[Noun]], fallback: dict[str, list[Noun]], key: str, rng: random.Random) -> Noun:
    return rng.choice(table.get(key) or fallback[key])


def _yes_no(about: str, yes: bool, noun: Noun, rng: random.Random, wording: Wording) -> Question:
    negated = rng.random() < NEGATED
    template = rng.choice(wording.absent if negated else wording.present)
    right = yes != negated
    return Question(YES_NO, ask(template, noun), [YES, NO], [right, not right], [True, True], [about], negated, template)


def summary(group: str, sample: dict, rng: random.Random, wording: Wording) -> Question:
    """Only for STCray's clean bags and their inserted-threat copies, where the whole bag is known."""
    yes = any(i["group"] == group for i in sample["items"])
    return _yes_no(group, yes, _noun(wording.groups, GROUP_NOUNS, group, rng), rng, wording)


def item_yes_no(dataset: str, there: set[str], rng: random.Random, wording: Wording) -> Question | None:
    item = rng.choice(ASKABLE[dataset])
    if unclear(item, there):
        return None
    return _yes_no(item, item in there, _noun(wording.items, ITEMS, item, rng), rng, wording)


def which(kind: int, dataset: str, there: set[str], rng: random.Random, wording: Wording) -> Question | None:
    low, high = OPTION_COUNTS[kind]
    items = rng.sample(ASKABLE[dataset], min(rng.randint(low, high), len(ASKABLE[dataset])))
    words = [_noun(wording.items, ITEMS, i, rng).word for i in items]
    answer = [i in there for i in items]
    if kind == ANY_OF:
        scored = [not unclear(i, there) for i in items]
        template = rng.choice(wording.which_all)
        return Question(ANY_OF, template, words, answer, scored, items, template=template) if any(scored) else None
    if sum(answer) > 1 or any(unclear(i, there) for i in items):  # one_of promises a single right answer
        return None
    options, answer, about = words + [None], answer + [not any(answer)], items + [None]
    order = rng.sample(range(len(options)), len(options))
    template = rng.choice(wording.which)
    return Question(ONE_OF, template, [options[k] for k in order], [answer[k] for k in order],
                    [True] * len(options), [about[k] for k in order], template=template)


def everything(dataset: str, there: set[str], rng: random.Random, wording: Wording) -> Question:
    """Any-of over every item the dataset labels, in a random order."""
    items = rng.sample(ASKABLE[dataset], len(ASKABLE[dataset]))
    template = rng.choice(wording.which_all)
    return Question(ANY_OF, template, [_noun(wording.items, ITEMS, i, rng).word for i in items], [i in there for i in items],
                    [not unclear(i, there) for i in items], items, template=template)


def questions_for(sample: dict, rng: random.Random, wording: Wording = TRAINING,
                  count: int = QUESTIONS_PER_SCAN) -> list[Question]:
    questions = [summary(g, sample, rng, wording) for g in (WEAPON, RESTRICTED)] if sample["pair"] else []
    if asks_about_items(sample):
        there = present(sample)
        questions.append(everything(sample["dataset"], there, rng, wording))
        for _ in range(count * TRIES):
            if len(questions) >= count:
                break
            kind = rng.choices(list(KINDS), weights=list(KINDS.values()))[0]
            if kind == YES_NO:
                question = item_yes_no(sample["dataset"], there, rng, wording)
            else:
                question = which(kind, sample["dataset"], there, rng, wording)
            if question:
                questions.append(question)
    return questions


def question_batch(per_scan: list[list[Question]], tokenizer: Tokenizer) -> dict:
    """Padded tensors for Kev: (scans, questions, tokens) and (scans, questions, options, tokens)."""
    scans, count = len(per_scan), max(len(qs) for qs in per_scan)
    options = max(len(q.options) for qs in per_scan for q in qs)
    texts = [[tokenizer.encode(q.text) for q in qs] for qs in per_scan]
    words = [[[tokenizer.encode(o) if o else [] for o in q.options] for q in qs] for qs in per_scan]
    tokens = max(len(t) for ts in texts for t in ts)
    option_tokens = max(1, max(len(w) for ws in words for q in ws for w in q))
    batch = {"ids": torch.full((scans, count, tokens), PAD), "type": torch.zeros(scans, count, dtype=torch.long),
             "valid": torch.zeros(scans, count, dtype=torch.bool),
             "option_ids": torch.full((scans, count, options, option_tokens), PAD)}
    for key in ("is_none", "option_valid", "answer", "scored"):
        batch[key] = torch.zeros(scans, count, options, dtype=torch.bool)
    for s, qs in enumerate(per_scan):
        for n, q in enumerate(qs):
            batch["ids"][s, n, : len(texts[s][n])] = torch.tensor(texts[s][n])
            batch["type"][s, n], batch["valid"][s, n] = q.type, True
            for o, option in enumerate(q.options):
                if words[s][n][o]:
                    batch["option_ids"][s, n, o, : len(words[s][n][o])] = torch.tensor(words[s][n][o])
                batch["is_none"][s, n, o] = option is None
                batch["option_valid"][s, n, o] = True
                batch["answer"][s, n, o], batch["scored"][s, n, o] = q.answer[o], q.scored[o]
    batch["mask"], batch["option_mask"] = batch["ids"] != PAD, batch["option_ids"] != PAD
    return batch


def main() -> None:
    from kev.data.curate import read
    from kev.data.loader import ScanDataset, load_library

    rng = random.Random(0)
    data = ScanDataset(read(), "train", train=False, library=load_library("val"))
    picks = [k for k, (i, inserted) in enumerate(data.entries) if inserted][:1]
    for name in ("pidray", "dvxray", "iedxray", "stcray"):
        picks += rng.sample([k for k, (i, _) in enumerate(data.entries) if data.lines[i]["dataset"] == name and data.lines[i]["items"]], 1)
    for k in picks:
        sample = {**data.lines[data.entries[k][0]], **{key: v for key, v in data[k].items() if key != "views"}}
        print(f"\n{sample['dataset']}: {', '.join(sorted(present(sample))) or 'nothing'}{' (inserted)' if sample['inserted'] else ''}")
        for q in questions_for(sample, rng):
            shown = "  ".join(f"[{'x' if a else ' '}]{'' if s else '?'} {o or 'none of these'}" for o, a, s in zip(q.options, q.answer, q.scored))
            print(f"  {['yes/no', 'one of', 'any of'][q.type]:7} {q.text!r:52} {shown}")


if __name__ == "__main__":
    main()
