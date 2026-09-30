"""Trains kev to answer questions about scans. The pretrained image encoder keeps learning slowly, the text encoder
stays frozen, and the readers and question decoder learn from scratch.

    uv run python -m kev.train.questions              (trains, resuming if runs/questions/checkpoint.pt exists)
    uv run python -m kev.train.questions --steps 300  (a short timed trial that saves nothing)
"""

import argparse
import json
import math
import random
import time
import zlib
from collections import Counter

import torch
import torch.nn.functional as F

from kev.data.curate import read
from kev.data.loader import BucketSampler, ScanDataset, collate, load_library
from kev.data.sources import DATA
from kev.device import pick_device
from kev.model.image import ImageEncoder
from kev.model.kev import ANY_OF, QUESTION_TYPES, YES_NO, Kev
from kev.model.text import TextEncoder
from kev.pretrain.image import passes_since_gain
from kev.questions.generate import HELD_OUT, TRAINING, Question, Wording, asks_about_items, question_batch, questions_for
from kev.tokenizer.bpe import Tokenizer

RUN = DATA.parent / "runs" / "questions"
IMAGE_ENCODER = DATA.parent / "runs" / "image" / "encoder_best.pt"
TEXT_ENCODER = DATA.parent / "runs" / "text" / "encoder.pt"
BATCH = 8
PASSES = 10
NEW_LR = 3e-4  # readers and question decoder, learning from scratch
IMAGE_LR = 5e-5  # the pretrained image encoder: adapt without forgetting
WARMUP = 0.03
WEIGHT_DECAY = 0.05
WORKERS = 8
VALIDATION_SCANS = 1024
PATIENCE = 3
MIN_PASSES = 3
MIN_GAIN = 0.001
MEMORY_FRACTION = 0.55
CACHE_LIMIT = 14e9
LONG_BATCH = 1200  # patches per scan; the cache is emptied first, since these need the most room
SCAN_KEYS = ("patches", "row", "col", "view", "valid")

_tokenizer = None


class Batches:
    """Scans plus questions picked for them: fresh each time, or the same every time when fixed."""

    def __init__(self, wording: Wording = TRAINING, fixed: bool = False):
        self.wording = wording
        self.fixed = fixed

    def questions(self, samples: list[dict]) -> tuple[dict, list[list[Question]]]:
        global _tokenizer
        _tokenizer = _tokenizer or Tokenizer.load()
        asked = []
        for s in samples:
            rng = random.Random(zlib.crc32(f"{s['id']}:{s['inserted']}".encode())) if self.fixed else random.Random()
            asked.append(questions_for(s, rng, self.wording))
        return question_batch(asked, _tokenizer), asked

    def __call__(self, samples: list[dict]) -> dict:
        batch = collate(samples)
        batch["questions"], batch["asked"] = self.questions(samples)
        return batch


def judged(q: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """What the loss counts: each one-answer question once, and each scored option of an any-of question once."""
    single = q["valid"] & (q["type"] != ANY_OF)
    options = q["scored"] & q["option_valid"] & (q["valid"] & (q["type"] == ANY_OF))[..., None]
    return single, options


def question_loss(kev: Kev, scores: torch.Tensor, q: dict) -> torch.Tensor:
    """Mean over judgements: cross-entropy for one-answer questions, a yes/no loss for each any-of option. Counting
    options one by one keeps a present item's "yes" in a long list from being diluted."""
    single, options = judged(q)
    masked = scores.masked_fill(~q["option_valid"], float("-inf"))
    single_loss = F.cross_entropy(masked[single], q["answer"][single].float().argmax(dim=-1), reduction="sum")
    each = F.binary_cross_entropy_with_logits(scores + kev.any_of_bias, q["answer"].float(), reduction="none")
    return (single_loss + each[options].sum()) / (single.sum() + options.sum())


def correct(kev: Kev, scores: torch.Tensor, q: dict) -> torch.Tensor:
    """Per question: one-answer questions need the top option right; any-of needs every scored option right."""
    p = kev.probabilities(scores, q)
    single = p.argmax(dim=-1) == q["answer"].float().argmax(dim=-1)
    each = ((p > 0.5) == q["answer"]) | ~(q["scored"] & q["option_valid"])
    return torch.where(q["type"] == ANY_OF, each.all(dim=-1), single) & q["valid"]


def nothing_there(question: Question) -> bool:
    """Would answering as if the bag were empty get this right?"""
    if question.type == YES_NO:
        return question.answer[1] != question.negated
    return all(not a for a, s, o in zip(question.answer, question.scored, question.options) if s and o is not None)


def presence(question: Question, p: torch.Tensor) -> list[tuple[bool, bool]]:
    """(is the item there, does kev say so) for each item a question judges: yes/no, and any-of's scored options."""
    if question.type == YES_NO:
        return [(question.answer[0] != question.negated, bool(p[0] > 0.5) != question.negated)]
    if question.type == ANY_OF:
        return [(a, bool(p[k] > 0.5)) for k, (a, s) in enumerate(zip(question.answer, question.scored)) if s]
    return []


def scores_for(kev: Kev, batch: dict, device: torch.device) -> tuple[torch.Tensor, dict]:
    scan = {k: batch[k].to(device) for k in SCAN_KEYS}
    q = {k: v.to(device) for k, v in batch["questions"].items()}
    return kev(scan, q), q


@torch.no_grad()
def validate(kev: Kev, batches: list[dict], device: torch.device) -> dict:
    kev.eval()
    loss, judgements, total, right, empty = 0.0, 0, Counter(), Counter(), Counter()
    there, found, absent, alarms = Counter(), Counter(), Counter(), Counter()
    for b in batches:
        scores, q = scores_for(kev, b, device)
        count = sum(x.sum().item() for x in judged(q))
        loss += question_loss(kev, scores, q).item() * count
        judgements += count
        ok = correct(kev, scores, q).cpu()
        p = kev.probabilities(scores, q).cpu()
        for s, (sample, asked) in enumerate(zip(b["samples"], b["asked"])):
            for n, question in enumerate(asked):
                for key in ("all", QUESTION_TYPES[question.type], sample["dataset"]):
                    total[key] += 1
                    right[key] += int(ok[s, n])
                    empty[key] += nothing_there(question)
                for is_there, says_so in presence(question, p[s, n]):
                    for key in ("all", sample["dataset"]):
                        there[key], found[key] = there[key] + is_there, found[key] + (is_there and says_so)
                        absent[key], alarms[key] = absent[key] + (not is_there), alarms[key] + (not is_there and says_so)
    kev.train()
    return {"loss": loss / judgements, "accuracy": {k: right[k] / total[k] for k in total},
            "nothing_there": {k: empty[k] / total[k] for k in total},
            "found": {k: found[k] / there[k] for k in there if there[k]},
            "false_alarms": {k: alarms[k] / absent[k] for k in absent if absent[k]}}


def learn(kev: Kev, optimizer: torch.optim.Optimizer, batch: dict, device: torch.device):
    """One training step. If the GPU runs out of memory, empty its cache and try once more; None if that fails too."""
    for attempt in range(2):
        try:
            scores, q = scores_for(kev, batch, device)
            loss = question_loss(kev, scores, q)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in kev.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            return loss, scores, q
        except RuntimeError as error:
            if "out of memory" not in str(error) or device.type != "mps":
                raise
            scores = q = loss = None
            optimizer.zero_grad(set_to_none=True)
            torch.mps.empty_cache()
    return None


def build(device: torch.device) -> Kev:
    image = ImageEncoder()
    image.load_state_dict(torch.load(IMAGE_ENCODER, map_location="cpu", weights_only=True)["model"])
    saved = torch.load(TEXT_ENCODER, map_location="cpu", weights_only=True)
    text = TextEncoder(saved["vocab"])
    text.load_state_dict(saved["model"])
    return Kev(image, text).to(device)


def learning_rate(step: int, total: int) -> float:
    """A fraction of each group's rate: a linear warm-up, then a cosine curve down to a tenth."""
    warm = max(1, int(WARMUP * total))
    if step < warm:
        return (step + 1) / warm
    return 0.1 + 0.45 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def optimizer_for(kev: Kev) -> torch.optim.Optimizer:
    groups = []
    for params, lr in ((list(kev.image.parameters()), IMAGE_LR),
                       ([p for n, p in kev.named_parameters() if p.requires_grad and not n.startswith("image.")], NEW_LR)):
        groups.append({"params": [p for p in params if p.dim() >= 2], "weight_decay": WEIGHT_DECAY, "base_lr": lr})
        groups.append({"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0, "base_lr": lr})
    return torch.optim.AdamW(groups, lr=NEW_LR, betas=(0.9, 0.95))


def train(trial_steps: int | None = None) -> None:
    device = pick_device()
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(MEMORY_FRACTION)
    torch.manual_seed(0)
    lines = [l for l in read() if asks_about_items(l)]
    data = ScanDataset(lines, "train", train=True, library=load_library("train"))
    sampler = BucketSampler([data.size(k) for k in range(len(data))], BATCH, zoom=True)
    loader = torch.utils.data.DataLoader(data, batch_sampler=sampler, collate_fn=Batches(), num_workers=WORKERS,
                                         persistent_workers=True)
    val = ScanDataset(lines, "val", train=False, library=load_library("val"))
    picks = sorted(random.Random(0).sample(range(len(val)), VALIDATION_SCANS))
    checks = {"training wording": [], "held-out wording": []}
    for b in BucketSampler([val.size(k) for k in picks], BATCH, shuffle=False):
        samples = [val[picks[i]] for i in b]
        scans = collate(samples)
        for name, wording in (("training wording", TRAINING), ("held-out wording", HELD_OUT)):
            questions, asked = Batches(wording, fixed=True).questions(samples)
            checks[name].append({**scans, "questions": questions, "asked": asked})

    kev = build(device)
    optimizer = optimizer_for(kev)
    total = PASSES * len(sampler)
    first_pass, step, val_losses = 0, 0, []
    RUN.mkdir(parents=True, exist_ok=True)
    checkpoint = RUN / "checkpoint.pt"
    if trial_steps is None and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        kev.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        first_pass, step, val_losses = saved["pass"] + 1, saved["step"], saved["val_losses"]
        print(f"resuming after pass {first_pass} (step {step:,})")
    print(f"{len(data):,} training scans, {total:,} steps over {PASSES} passes, {VALIDATION_SCANS} validation scans",
          flush=True)

    with open(RUN / "log.jsonl", "a") as log:
        def record(entry: dict) -> None:
            if trial_steps is None:
                log.write(json.dumps(entry) + "\n")
                log.flush()
            shown = {k: v for k, v in entry.items() if not isinstance(v, dict)}
            print("  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in shown.items()), flush=True)

        def check(n: int) -> None:
            results = {name: validate(kev, batches, device) for name, batches in checks.items()}
            val_losses.append(results["training wording"]["loss"])
            record({"pass": n, "val_loss": val_losses[-1], "accuracy": results["training wording"]["accuracy"]["all"],
                    "held_out_accuracy": results["held-out wording"]["accuracy"]["all"], "results": results})
            for name, r in results.items():
                print(f"  {name}: " + "  ".join(f"{k} {r['accuracy'][k]:.3f} (empty-bag guess {r['nothing_there'][k]:.3f})"
                                              for k in sorted(r["accuracy"])), flush=True)
                print(f"    items there that kev reports: " + "  ".join(f"{k} {v:.3f}" for k, v in sorted(r["found"].items()))
                      + "   false alarms: " + "  ".join(f"{k} {v:.3f}" for k, v in sorted(r["false_alarms"].items())), flush=True)

        if first_pass == 0 and trial_steps is None:
            check(0)
        start, seen, running = time.perf_counter(), 0, []
        for n in range(first_pass, PASSES):
            sampler.epoch = n
            for batch in loader:
                fraction = learning_rate(step, total)
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * fraction
                if device.type == "mps" and batch["patches"].shape[1] > LONG_BATCH:
                    torch.mps.empty_cache()
                result = learn(kev, optimizer, batch, device)
                if result is None:
                    record({"step": step, "skipped_batch": "out of memory twice", "patches": batch["patches"].shape[1]})
                    continue
                loss, scores, q = result
                if device.type == "mps" and torch.mps.driver_allocated_memory() > CACHE_LIMIT:
                    torch.mps.empty_cache()
                step += 1
                seen += len(batch["samples"])
                running.append((loss.detach(), correct(kev, scores.detach(), q).sum(), q["valid"].sum()))
                if step % 200 == 0 or step == trial_steps:
                    elapsed = time.perf_counter() - start
                    losses, rights, counts = (torch.stack(x).sum().item() for x in zip(*running))
                    record({"step": step, "loss": losses / len(running), "accuracy": rights / counts,
                            "new_lr": NEW_LR * fraction, "scans_per_s": round(seen / elapsed, 1),
                            "minutes": round(elapsed / 60, 1), "gpu_gb": round(torch.mps.driver_allocated_memory() / 1e9, 1)})
                    running = []
                if step == trial_steps:
                    print(f"about {len(sampler) * elapsed / step / 60:.0f} minutes a pass")
                    return
            check(n + 1)
            torch.save({"model": kev.state_dict(), "optimizer": optimizer.state_dict(), "pass": n, "step": step,
                        "val_losses": val_losses}, checkpoint)
            if val_losses[-1] == min(val_losses):
                torch.save({"model": kev.state_dict(), "pass": n + 1}, RUN / "kev_best.pt")
            if len(val_losses) - 1 >= MIN_PASSES and passes_since_gain(val_losses, MIN_GAIN) >= PATIENCE:
                record({"stopped_early_after_pass": n + 1})
                return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, help="a short timed trial that saves nothing")
    train(parser.parse_args().steps)


if __name__ == "__main__":
    main()
