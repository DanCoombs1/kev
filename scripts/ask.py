"""Ask kev about test scans it has never seen, or about your own X-ray image. Opens the scan, then answers whatever
you type. kev only knows dual-energy X-ray baggage scans on a white background (orange organic, green mixed, blue
metal); anything else gets confident nonsense.

    uv run python scripts/ask.py [--dataset stcray] [--seed 3]
    uv run python scripts/ask.py --image bag.png [--image side-view.png] [--like pidray]

    is there a knife in this bag?                     yes/no
    one: which of these is in the bag? | knife, gun   one of these, or none of them
    any: which of these are in the bag? | knife, gun  any number of them
    all                                               every item this dataset labels
    show                                              what's really in the bag
    new                                               another scan
    quit
"""

import argparse
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from kev.data.curate import read
from kev.data.loader import ScanDataset, collate
from kev.data.prepare import SCALE, prepare_image
from kev.device import pick_device
from kev.model.kev import ANY_OF, ONE_OF, YES_NO
from kev.questions.bank import NO, YES
from kev.questions.generate import ASKABLE, Question, question_batch
from kev.tokenizer.bpe import Tokenizer
from kev.train.questions import RUN, SCAN_KEYS, build

PICTURE = Path(__file__).resolve().parents[1] / "runs" / "explore" / "ask.png"
FONT = ImageFont.load_default(size=16)


def picture(sample: dict, boxes: bool) -> None:
    views = [Image.fromarray(v.permute(1, 2, 0).numpy()) for v in sample["views"]]
    if boxes:
        for v, image in enumerate(views):
            draw = ImageDraw.Draw(image)
            for item in sample["items"]:
                if item["view"] == v and item["box"]:
                    x, y, w, h = item["box"]
                    draw.rectangle([x, y, x + w, y + h], outline=(230, 0, 180), width=3)
                    draw.text((x + 3, y + 2), item["name"], fill=(230, 0, 180), font=FONT)
    sheet = Image.new("RGB", (sum(im.width for im in views) + 10 * len(views), max(im.height for im in views)), "white")
    x = 0
    for image in views:
        sheet.paste(image, (x, 0))
        x += image.width + 10
    PICTURE.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(PICTURE)


def own_scan(paths: list[str], like: str) -> dict:
    """Your image(s) prepared like a scan from the `like` dataset: cropped to the bag and scaled to its pixel size."""
    views = []
    for path in paths:
        with Image.open(path) as image:
            prepared = prepare_image(image.convert("RGB"), like, [])
        views.append(torch.from_numpy(np.asarray(prepared.image).copy()).permute(2, 0, 1))
    return {"id": ", ".join(paths), "dataset": like, "views": views, "items": []}


def parse(line: str, dataset: str) -> Question | None:
    blank = lambda n: dict(answer=[False] * n, scored=[True] * n, about=[None] * n)
    if line == "all":
        names = ASKABLE[dataset]
        return Question(ANY_OF, "which of these are in the bag?", names, **blank(len(names)))
    for prefix, kind in (("one:", ONE_OF), ("any:", ANY_OF)):
        if line.startswith(prefix):
            if "|" not in line:
                print("  put the options after a |, separated by commas")
                return None
            text, options = line[len(prefix):].split("|", 1)
            options = [o.strip() for o in options.split(",") if o.strip()]
            if kind == ONE_OF:
                options.append(None)
            return Question(kind, text.strip(), options, **blank(len(options)))
    return Question(YES_NO, line, [YES, NO], **blank(2))


def answer(kev, tokenizer, scan: dict, question: Question, device) -> tuple[list[float], float]:
    q = {k: v.to(device) for k, v in question_batch([[question]], tokenizer).items()}
    start = time.perf_counter()
    with torch.no_grad():
        p = kev.probabilities(kev(scan, q), q)[0, 0, : len(question.options)].tolist()
    if device.type == "mps":
        torch.mps.synchronize()
    return p, time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(SCALE), help="only scans from this dataset")
    parser.add_argument("--seed", type=int, help="pick the same scans again")
    parser.add_argument("--no-open", action="store_true", help="save the picture without opening it")
    parser.add_argument("--image", action="append", help="your own X-ray image; give it twice for a top and side view")
    parser.add_argument("--like", choices=sorted(SCALE), default="stcray",
                        help="which dataset your image looks like: sets its pixel scale and the items `all` asks about")
    args = parser.parse_args()
    if args.image and len(args.image) > 2:
        parser.error("at most two views")

    device = pick_device()
    kev = build(device)
    saved = torch.load(RUN / "kev_best.pt", map_location=device, weights_only=True)
    kev.load_state_dict(saved["model"])
    kev.eval()
    tokenizer = Tokenizer.load()
    rng = random.Random(args.seed)
    if args.image:
        print(f"kev from pass {saved['pass']}, your image prepared like a {args.like} scan. Type a question, or: all, quit")
    else:
        lines = [l for l in read() if args.dataset in (None, l["dataset"])]
        test = ScanDataset(lines, "test", train=False)
        print(f"kev from pass {saved['pass']}, {len(test):,} test scans it has never seen. Type a question, or: all, show, new, quit")

    sample = scan = None
    while True:
        if sample is None:
            sample = own_scan(args.image, args.like) if args.image else test[rng.randrange(len(test))]
            scan = {k: v.to(device) for k, v in collate([sample]).items() if k in SCAN_KEYS}
            picture(sample, boxes=False)
            if not args.no_open:
                subprocess.run(["open", str(PICTURE)])
            answer(kev, tokenizer, scan, parse("all", sample["dataset"]), device)  # the first pass builds GPU kernels
            print(f"\n{sample['dataset']} scan {sample['id']} ({len(sample['views'])} view{'s' * (len(sample['views']) > 1)}) -> {PICTURE}")
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if line in ("quit", "exit", "q"):
            break
        if line in ("new", "show") and args.image:
            print("  that's for test scans; your own image has no labels")
            continue
        if line == "new":
            sample = None
            continue
        if line == "show":
            names = sorted({i["name"] for i in sample["items"]})
            print(f"  really in the bag: {', '.join(names) or 'nothing labelled'} (boxes drawn on the picture)")
            picture(sample, boxes=True)
            if not args.no_open:
                subprocess.run(["open", str(PICTURE)])
            continue
        question = parse(line, sample["dataset"]) if line else None
        if question is None:
            continue
        p, seconds = answer(kev, tokenizer, scan, question, device)
        ranked = sorted(zip(question.options, p), key=lambda op: -op[1]) if question.type != YES_NO else zip(question.options, p)
        print("  " + "   ".join(f"{o or 'none of these'} {v:.0%}" for o, v in ranked) + f"   ({seconds * 1000:.0f} ms)")


if __name__ == "__main__":
    main()
