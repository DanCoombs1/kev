"""Turn the manifest (the datasets exactly as published) into kev's cleaned dataset, data/curated.jsonl.

Step 3b: identical copies of the same scan (found by kev.data.duplicates) are merged into one
scan carrying the union of every copy's labels, because some datasets split one scan into
several copies that each label only some of its items. When copies sit on both sides of the
train/test line, the test copy survives, so training never holds a copy of a test scan.

Step 3c: every scan gets a group (scans so alike that they must stay on one side of the line)
and a role: train, val or test.

    uv run python -m kev.data.curate
"""

import hashlib
import json
from collections import Counter

from kev.data.duplicates import DUPLICATES
from kev.data.manifest import read as read_manifest
from kev.data.sources import DATA, scan_format

CURATED = DATA / "curated.jsonl"
SAME_PICTURE = 1  # fingerprint answers that may differ; beyond this, matches include different objects (Step 3a)
SAME_ITEM_IOU = 0.5  # two boxes of the same name overlapping this much are one item labelled twice


def iou(a: list[float], b: list[float]) -> float:
    """Overlap of two boxes divided by the area they cover together: 1 = identical, 0 = apart."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return overlap / (aw * ah + bw * bh - overlap)


def copy_groups(ids: list[str]) -> list[list[str]]:
    """Group scans that are copies of each other, following chains (A copies B, B copies C)."""
    parent = {i: i for i in ids}

    def root(i: str) -> str:  # union-find: follow parents up to the group's representative
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for pair in json.loads(DUPLICATES.read_text()):
        if pair["different"] <= SAME_PICTURE:
            parent[root(pair["a"])] = root(pair["b"])
    groups = {}
    for i in ids:
        groups.setdefault(root(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def keep_priority(line: dict) -> tuple:
    return (
        line["split"].startswith("test"),  # a test copy wins, so train never holds a copy of a test scan
        line["dataset"] == "stcray",  # the clean bags STCray and IEDXray both published: keep STCray's
        len(line["items"]),  # the most completely labelled copy
        line["id"],  # any fixed order, so re-running gives the same result
    )


def merge(keep: dict, copies: list[dict]) -> dict:
    """The surviving copy, plus any item another copy labelled that it doesn't already have."""
    keep = {**keep, "items": list(keep["items"]), "copies": [c["id"] for c in copies]}
    for copy in copies:
        if copy["sizes"] != keep["sizes"]:
            continue  # boxes are only comparable between pictures of the same size
        for item in copy["items"]:
            already = any(
                i["name"] == item["name"] and i["view"] == item["view"] and
                (i["box"] is None or item["box"] is None or iou(i["box"], item["box"]) >= SAME_ITEM_IOU)
                for i in keep["items"]
            )
            if not already:
                keep["items"].append(item)
    return keep


def group_of(line: dict) -> str:
    """The name of the group a scan belongs to: scans in one group always share a role."""
    meta = line["meta"]
    if line["dataset"] == "stcray" and "variant" in meta:
        # One packing: the same object in the same bag and arrangement, shot from several angles and repeatedly.
        # STCray restarts its arrangement numbers in each split (Wrench1_B4_L1_C1 in train and in test are
        # different packings, checked by eye), so a packing only means something within its split.
        return f"stcray|{line['split']}|" + "|".join(str(meta.get(k)) for k in ("variant", "B", "L", "C", "other"))
    if line["dataset"] == "compass_xp":
        return f"compass_xp|{line['items'][0]['label']}|{meta['instance']}"  # one physical object, all its poses
    return line["id"]  # DvXray: one bag, both views on one line. PIDray, IEDXray: nothing links scans.


def lottery(group: str) -> int:
    """A number from 0 to 99 that looks random but is always the same for the same group.

    SHA-256 rather than Python's hash(), which changes every time Python starts.
    """
    return int(hashlib.sha256(group.encode()).hexdigest(), 16) % 100


def role_of(line: dict) -> str:
    ticket = lottery(line["group"])
    if line["split"] == "all":  # DvXray and COMPASS-XP ship without a test set: make our own
        return "test" if ticket < 20 else "val" if ticket < 30 else "train"
    if line["split"].startswith("test"):
        return "test"  # official test sets stay untouched
    return "val" if ticket < 10 else "train"


def curate(manifest: list[dict]) -> list[dict]:
    by_id = {line["id"]: line for line in manifest}
    replaced = {}  # id of every copy that doesn't survive -> None; survivor id -> merged line
    for group in copy_groups(list(by_id)):
        lines = sorted((by_id[i] for i in group), key=keep_priority, reverse=True)
        replaced[lines[0]["id"]] = merge(lines[0], lines[1:])
        for dropped in lines[1:]:
            replaced[dropped["id"]] = None
    curated = []
    for line in manifest:
        line = replaced.get(line["id"], line)
        if line is None:
            continue
        line = {"copies": [], **line}
        # PIDray's tall-format trays were labelled one item type per copy (Step 3a), so a tray may hold
        # items nobody labelled: only "yes" answers can be trusted on them.
        line["yes_only"] = line["dataset"] == "pidray" and scan_format(line["sizes"][0][1]) == "tall"
        line["group"] = group_of(line)
        line["role"] = role_of(line)
        curated.append(line)
    return curated


def read() -> list[dict]:
    with open(CURATED) as f:
        return [json.loads(line) for line in f]


def main() -> None:
    manifest = read_manifest()
    curated = curate(manifest)
    with open(CURATED, "w") as f:
        for line in curated:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    before = Counter((l["dataset"], l["split"]) for l in manifest)
    after = Counter((l["dataset"], l["split"]) for l in curated)
    published = {l["id"]: l for l in manifest}
    gained = Counter()
    for l in curated:
        if l["copies"]:
            gained[l["dataset"]] += len(l["items"]) - len(published[l["id"]]["items"])
    print(f"wrote {len(curated):,} scans to {CURATED.relative_to(DATA.parent)} (from {len(manifest):,})")
    print(f"{'dataset':<11} {'split':<12} {'published':>9} {'curated':>8} {'removed':>8}")
    for key in sorted(before):
        print(f"{key[0]:<11} {key[1]:<12} {before[key]:>9,} {after[key]:>8,} {before[key] - after[key]:>8,}")
    print("items recovered from copies:", dict(gained))
    print("scans marked yes-only:", sum(l["yes_only"] for l in curated))
    print(f"\n{'dataset':<11} {'train':>7} {'val':>6} {'test':>7}   groups")
    for dataset in sorted({l["dataset"] for l in curated}):
        lines = [l for l in curated if l["dataset"] == dataset]
        roles = Counter(l["role"] for l in lines)
        print(f"{dataset:<11} {roles['train']:>7,} {roles['val']:>6,} {roles['test']:>7,}   {len({l['group'] for l in lines}):,}")


if __name__ == "__main__":
    main()
