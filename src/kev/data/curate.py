"""Builds data/curated.jsonl from the manifest: merges duplicate scans and assigns train/val/test roles.

    uv run python -m kev.data.curate
"""

import hashlib
import json
from collections import Counter

from kev.data.duplicates import DUPLICATES
from kev.data.manifest import read as read_manifest
from kev.data.sources import DATA, scan_format

CURATED = DATA / "curated.jsonl"
SAME_PICTURE = 1  # dHash bits; at 7, COMPASS-XP matches different objects on the same tray
SAME_ITEM_IOU = 0.5


def iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return overlap / (aw * ah + bw * bh - overlap)


def copy_groups(ids: list[str]) -> list[list[str]]:
    parent = {i: i for i in ids}

    def root(i: str) -> str:
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
        line["split"].startswith("test"),  # never keep a train copy of a test scan
        line["dataset"] == "stcray",  # STCray and IEDXray publish the same clean bags
        len(line["items"]),
        line["id"],
    )


def merge(keep: dict, copies: list[dict]) -> dict:
    """Keep one copy, adding items that only the other copies labelled."""
    keep = {**keep, "items": list(keep["items"]), "copies": [c["id"] for c in copies]}
    for copy in copies:
        if copy["sizes"] != keep["sizes"]:
            continue
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
    meta = line["meta"]
    if line["dataset"] == "stcray" and "variant" in meta:
        # STCray reuses packing codes across splits, so a packing is only unique within its split.
        return f"stcray|{line['split']}|" + "|".join(str(meta.get(k)) for k in ("variant", "B", "L", "C", "other"))
    if line["dataset"] == "compass_xp":
        return f"compass_xp|{line['items'][0]['label']}|{meta['instance']}"
    return line["id"]


def bucket(group: str) -> int:
    # sha256 rather than hash(), which is salted per process
    return int(hashlib.sha256(group.encode()).hexdigest(), 16) % 100


def role_of(line: dict) -> str:
    b = bucket(line["group"])
    if line["split"] == "all":
        return "test" if b < 20 else "val" if b < 30 else "train"
    if line["split"].startswith("test"):
        return "test"
    return "val" if b < 10 else "train"


def curate(manifest: list[dict]) -> list[dict]:
    by_id = {line["id"]: line for line in manifest}
    replaced = {}
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
        # PIDray trays were labelled one item type per copy, so they may hold unlabelled items.
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
    print("yes-only scans:", sum(l["yes_only"] for l in curated))
    print(f"\n{'dataset':<11} {'train':>7} {'val':>6} {'test':>7}   groups")
    for dataset in sorted({l["dataset"] for l in curated}):
        lines = [l for l in curated if l["dataset"] == dataset]
        roles = Counter(l["role"] for l in lines)
        print(f"{dataset:<11} {roles['train']:>7,} {roles['val']:>6,} {roles['test']:>7,}   {len({l['group'] for l in lines}):,}")


if __name__ == "__main__":
    main()
