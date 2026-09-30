"""Can kev, starting from both pretrained encoders, memorise the answers for a handful of scans? If it can't,
something is broken.

    uv run python scripts/check_questions.py
"""

import random
import time
from itertools import batched

import torch

from kev.data.curate import read
from kev.data.loader import ScanDataset, load_library
from kev.device import pick_device
from kev.questions.generate import asks_about_items
from kev.train.questions import BATCH, MEMORY_FRACTION, Batches, build, correct, optimizer_for, question_loss, scores_for

PER_DATASET = 3
ROUNDS = 300


def accuracy(kev, batches, device) -> float:
    kev.eval()
    with torch.no_grad():
        right = total = 0
        for b in batches:
            scores, q = scores_for(kev, b, device)
            right += correct(kev, scores, q).sum().item()
            total += q["valid"].sum().item()
    kev.train()
    return right / total


def main() -> None:
    device = pick_device()
    torch.mps.set_per_process_memory_fraction(MEMORY_FRACTION)
    torch.manual_seed(0)
    rng = random.Random(0)
    data = ScanDataset([l for l in read() if asks_about_items(l)], "train", train=False, library=load_library("train"))
    picks = []
    for name in ("pidray", "stcray", "dvxray", "iedxray"):
        picks += rng.sample([k for k, (i, _) in enumerate(data.entries) if data.lines[i]["dataset"] == name], PER_DATASET)
    inserted = rng.sample([k for k, (i, insert) in enumerate(data.entries) if insert], 2)
    picks += inserted + [k - 1 for k in inserted]  # both halves of two pairs
    batches = [Batches(fixed=True)([data[k] for k in chunk]) for chunk in batched(picks, BATCH)]
    questions = sum(b["questions"]["valid"].sum().item() for b in batches)

    kev = build(device)
    optimizer = optimizer_for(kev)
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"]
    print(f"{len(picks)} scans, {questions} questions; before training {accuracy(kev, batches, device):.0%} right")
    start = time.perf_counter()
    for n in range(1, ROUNDS + 1):
        for b in batches:
            scores, q = scores_for(kev, b, device)
            loss = question_loss(kev, scores, q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if n % 50 == 0:
            print(f"round {n:3d}  loss {loss.item():.3f}  {accuracy(kev, batches, device):.0%} right")
    print(f"({time.perf_counter() - start:.0f}s)\n")

    kev.eval()
    b = batches[-1]
    with torch.no_grad():
        scores, q = scores_for(kev, b, device)
        p = kev.probabilities(scores, q).cpu()
    for s, (sample, asked) in enumerate(zip(b["samples"], b["asked"])):
        print(f"{sample['dataset']}{' (inserted)' if sample['inserted'] else ''}: {', '.join(i['name'] for i in sample['items']) or 'nothing'}")
        for n, question in enumerate(asked[:3]):
            shown = "  ".join(f"{'*' if a else ' '}{o or 'none'} {p[s, n, k]:.2f}" for k, (o, a) in enumerate(zip(question.options, question.answer)))
            print(f"  {question.text!r:48} {shown}")


if __name__ == "__main__":
    main()
