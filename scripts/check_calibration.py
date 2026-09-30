"""Is kev's confidence honest? Reliability diagrams and expected calibration error (ECE) on validation scans the
training never checks, before and after temperature scaling (one number, fitted on half the scans, measured on the
other half). Writes calibration.png in runs/questions/.

    uv run python scripts/check_calibration.py [runs/questions/kev_best.pt]
"""

import random
import sys
from itertools import batched
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from kev.data.curate import read
from kev.data.loader import ScanDataset, load_library
from kev.device import pick_device
from kev.model.kev import ANY_OF, YES_NO
from kev.questions.generate import TRAINING, asks_about_items
from kev.train.questions import BATCH, RUN, VALIDATION_SCANS, Batches, build, scores_for

SCANS = 2048
BINS = 10
MIN_DRAWN = 20  # bins with fewer predictions than this are too noisy to plot
TEMPERATURES = [0.5 + 0.05 * n for n in range(51)]
FONT = ImageFont.load_default(size=14)


def collect(kev, val, picks, device) -> dict:
    """Raw scores for each item judgement ("is it there?") and each one-answer question."""
    out = {"item_scores": [], "item_there": [], "single_scores": [], "single_answer": []}
    for chunk in batched(picks, BATCH):
        batch = Batches(TRAINING, fixed=True)([val[k] for k in chunk])
        with torch.no_grad():
            scores, q = scores_for(kev, batch, device)
        scores = scores.cpu()
        for s, asked in enumerate(batch["asked"]):
            for n, question in enumerate(asked):
                row = scores[s, n, : len(question.options)]
                if question.type == ANY_OF:
                    for k, (answer, scored) in enumerate(zip(question.answer, question.scored)):
                        if scored:
                            out["item_scores"].append(row[k] + kev.any_of_bias.item())
                            out["item_there"].append(answer)
                else:
                    out["single_scores"].append(row)
                    out["single_answer"].append(question.answer.index(True))
                    if question.type == YES_NO:  # also an item judgement: the logit of "it's there"
                        there_is_yes = not question.negated
                        logit = (row[0] - row[1]) if there_is_yes else (row[1] - row[0])
                        out["item_scores"].append(logit)
                        out["item_there"].append(question.answer[0] == there_is_yes)
    return out


def item_probabilities(data: dict, indices: list[int], t: float) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.stack([torch.as_tensor(data["item_scores"][i]) for i in indices])
    there = torch.tensor([data["item_there"][i] for i in indices], dtype=torch.float)
    return torch.sigmoid(scores / t), there


def top_choices(data: dict, indices: list[int], t: float) -> tuple[torch.Tensor, torch.Tensor]:
    confidence, right = [], []
    for i in indices:
        p = (data["single_scores"][i] / t).softmax(dim=-1)
        confidence.append(p.max())
        right.append(float(p.argmax() == data["single_answer"][i]))
    return torch.stack(confidence), torch.tensor(right)


def ece(confidence: torch.Tensor, outcome: torch.Tensor) -> tuple[float, list[tuple[float, float, int]]]:
    bins, total = [], 0.0
    for b in range(BINS):
        inside = (confidence >= b / BINS) & ((confidence < (b + 1) / BINS) | (b == BINS - 1))
        if inside.any():
            conf, freq, n = confidence[inside].mean().item(), outcome[inside].mean().item(), int(inside.sum())
            bins.append((conf, freq, n))
            total += n / len(confidence) * abs(conf - freq)
    return total, bins


def log_loss(p: torch.Tensor, outcome: torch.Tensor) -> float:
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(outcome * p.log() + (1 - outcome) * (1 - p).log()).mean().item()


def panel(draw: ImageDraw.ImageDraw, x0: int, title: str, curves: list[tuple[str, list, tuple]]) -> None:
    size, y0 = 360, 40
    draw.text((x0, 10), title, fill="black", font=FONT)
    draw.rectangle([x0, y0, x0 + size, y0 + size], outline="black")
    draw.line([x0, y0 + size, x0 + size, y0], fill=(180, 180, 180), width=1)  # perfect calibration
    for n, (name, bins, colour) in enumerate(curves):
        points = [(x0 + conf * size, y0 + size - freq * size) for conf, freq, n in bins if n >= MIN_DRAWN]
        if len(points) > 1:
            draw.line(points, fill=colour, width=3)
        for x, y in points:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=colour)
        draw.text((x0 + 10, y0 + 10 + 18 * n), name, fill=colour, font=FONT)
    draw.text((x0, y0 + size + 8), "confidence (x) against how often it was right (y)", fill="black", font=FONT)


def main() -> None:
    device = pick_device()
    torch.mps.set_per_process_memory_fraction(0.3)
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RUN / "kev_best.pt"
    kev = build(device)
    saved = torch.load(path, map_location=device, weights_only=True)
    kev.load_state_dict(saved["model"])
    kev.eval()
    val = ScanDataset([l for l in read() if asks_about_items(l)], "val", train=False, library=load_library("val"))
    checked = set(random.Random(0).sample(range(len(val)), VALIDATION_SCANS))
    unused = [k for k in range(len(val)) if k not in checked]
    picks = random.Random(1).sample(unused, min(SCANS, len(unused)))
    data = collect(kev, val, picks, device)
    print(f"{path}, pass {saved['pass']}: {len(picks)} validation scans the training never checks, "
          f"{len(data['item_there']):,} item judgements, {len(data['single_answer']):,} one-answer questions\n")

    rng = random.Random(2)
    results = {}
    for name, count, measure in (("item there?", len(data["item_there"]), item_probabilities),
                                 ("one-answer top choice", len(data["single_answer"]), top_choices)):
        order = rng.sample(range(count), count)
        fit, test = order[: count // 2], order[count // 2:]
        best = min(TEMPERATURES, key=lambda t: log_loss(*measure(data, fit, t)))
        raw_ece, raw_bins = ece(*measure(data, test, 1.0))
        scaled_ece, scaled_bins = ece(*measure(data, test, best))
        results[name] = (raw_bins, scaled_bins, best)
        print(f"{name}: ECE {raw_ece:.3f} as trained, {scaled_ece:.3f} with temperature {best:.2f} (measured on the other half)")
        for (conf, freq, n) in raw_bins:
            print(f"    says {conf:5.1%}  right {freq:5.1%}  ({n:,})")

    sheet = Image.new("RGB", (800, 440), "white")
    draw = ImageDraw.Draw(sheet)
    for x0, (name, (raw, scaled, t)) in zip((20, 420), results.items()):
        panel(draw, x0, name, [("as trained", raw, (200, 60, 60)), (f"temperature {t:.2f}", scaled, (40, 90, 200))])
    sheet.save(RUN / "calibration.png")
    print(f"\n-> {RUN / 'calibration.png'}")


if __name__ == "__main__":
    main()
