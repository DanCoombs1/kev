"""How a saved kev does on each sentence it's asked with, in training and held-out wording, next to the empty-bag
guess, and how often it reports items that are there when they're named by a held-out synonym. Uses the same validation
scans and questions the training checks.

    uv run python scripts/check_wording.py [runs/questions/kev_best.pt]
"""

import random
import sys
from collections import defaultdict
from itertools import batched
from pathlib import Path

import torch

from kev.data.curate import read
from kev.data.loader import ScanDataset, load_library
from kev.device import pick_device
from kev.model.kev import ANY_OF, QUESTION_TYPES
from kev.questions.bank import ITEMS_HELD_OUT
from kev.questions.generate import HELD_OUT, TRAINING, asks_about_items
from kev.train.questions import BATCH, RUN, VALIDATION_SCANS, Batches, build, correct, nothing_there, scores_for


def main() -> None:
    device = pick_device()
    torch.mps.set_per_process_memory_fraction(0.3)
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RUN / "kev_best.pt"
    kev = build(device)
    saved = torch.load(path, map_location=device, weights_only=True)
    kev.load_state_dict(saved["model"])
    kev.eval()
    print(f"{path}, pass {saved['pass']}\n")

    val = ScanDataset([l for l in read() if asks_about_items(l)], "val", train=False, library=load_library("val"))
    picks = sorted(random.Random(0).sample(range(len(val)), VALIDATION_SCANS))
    found = {}
    for name, wording in (("training wording", TRAINING), ("held-out wording", HELD_OUT)):
        tally = defaultdict(lambda: [0, 0, 0])  # right, total, empty-bag guess right
        seen = defaultdict(lambda: [0, 0])  # (any-of sentence) or (item, word): items there that kev reports, items there
        for chunk in batched(picks, BATCH):
            batch = Batches(wording, fixed=True)([val[k] for k in chunk])
            with torch.no_grad():
                scores, q = scores_for(kev, batch, device)
                ok = correct(kev, scores, q).cpu()
                p = kev.probabilities(scores, q).cpu()
            for s, asked in enumerate(batch["asked"]):
                for n, question in enumerate(asked):
                    t = tally[(QUESTION_TYPES[question.type], question.template)]
                    t[0] += int(ok[s, n])
                    t[1] += 1
                    t[2] += nothing_there(question)
                    if question.type == ANY_OF:
                        for k, (item, word, answer, scored) in enumerate(zip(question.about, question.options, question.answer, question.scored)):
                            if scored and answer:
                                for key in (("sentence", question.template), ("word", item, word)):
                                    seen[key][0] += int(p[s, n, k] > 0.5)
                                    seen[key][1] += 1
        found[name] = seen
        print(f"{name}: yes/no sentences, worst first (kev, empty-bag guess, questions)")
        rows = [(r / n - e / n, kind, template, r / n, e / n, n) for (kind, template), (r, n, e) in tally.items() if kind == "yes_no"]
        for gap, _, template, acc, empty, n in sorted(rows):
            print(f"  {acc:5.1%} {empty:5.1%} {n:4d}  {gap:+6.1%}  {template}")
        print(f"{name}: any-of sentences, items there that kev reports")
        for key, (hit, n) in sorted(found[name].items(), key=lambda kv: kv[1][0] / kv[1][1]):
            if key[0] == "sentence":
                print(f"  {hit / n:5.1%} of {n:4d}  {key[1]}")
        print()

    print("items with a held-out synonym: reported when named by a training word vs by the held-out word")
    held_words = {(item, n.word) for item, nouns in ITEMS_HELD_OUT.items() for n in nouns}
    for item in sorted(ITEMS_HELD_OUT):
        train = [v for k, v in found["training wording"].items() if k[0] == "word" and k[1] == item]
        held = [v for k, v in found["held-out wording"].items() if k[0] == "word" and (k[1], k[2]) in held_words and k[1] == item]
        if train and held:
            t_hit, t_n = map(sum, zip(*train))
            h_hit, h_n = map(sum, zip(*held))
            word = next(w for i, w in held_words if i == item)
            print(f"  {item:26} training words {t_hit / t_n:5.1%} of {t_n:3d}   {word!r:28} {h_hit / h_n:5.1%} of {h_n:3d}")


if __name__ == "__main__":
    main()
