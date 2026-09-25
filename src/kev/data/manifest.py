"""The manifest: every scan from every dataset as one line of JSON, in one shape, with kev's item names.

Every later step reads data/manifest.jsonl instead of the five original formats. Build it with:

    uv run python -m kev.data.manifest
"""

import json
import re
from collections import Counter

from kev.data.names import group, kev_name
from kev.data.sources import DATA, Scan, all_scans, image_size

MANIFEST = DATA / "manifest.jsonl"

# Licence tier per dataset, kept on every line so a commercially usable kev can be trained from "commercial" lines alone.
TIERS = {
    "stcray": ("commercial", "CC BY 4.0 (paper) / Apache-2.0 (Hugging Face)"),
    "iedxray": ("commercial", "CC BY 4.0"),
    "compass_xp": ("commercial", "CC BY 4.0"),
    "pidray": ("academic", "academic use only"),
    "dvxray": ("academic", "no licence stated; treated as academic use only"),
}


def stcray_staging(stem: str) -> dict:
    """Decode STCray's filename code, e.g. Gun4_B1_L10_C10_Loc1_phi1_th1_1 -> item variant Gun4, bag B1, ...

    The meaning of L, C, Loc, phi and th isn't documented, so they keep their filename letters,
    and values stay strings because some carry a letter (L4A). Some test scans are named by
    timestamp instead and carry no code; ~150 explosive scans use another scheme, kept verbatim.
    """
    if re.match(r"\d{4}-\d{2}-\d{2}", stem):
        return {}
    parts = stem.split("_")
    staging = {"variant": parts[0]}
    for part in parts[1:]:
        if part.isdigit():
            staging["shot"] = part  # the repeat number at the end
        elif m := re.fullmatch(r"(B|L|C|Loc|phi|th)(\d+[A-Z]?)", part):
            staging[m.group(1)] = m.group(2)
        else:
            staging["other"] = part  # e.g. "Explosive (Integrated) P-1 B-2 C-1 BAG-3 L-2"
    return staging


def clamp(box: tuple[float, float, float, float], size: tuple[int, int]) -> list[float]:
    """Trim a box to the image: 3 of ~157k boxes stick out by up to 4 px (Step 2c)."""
    x, y, w, h = box
    width, height = size
    x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
    return [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]


def to_line(scan: Scan) -> dict | None:
    """One manifest line, or None for a scan kev must not use."""
    if scan.dataset == "pidray" and not scan.items:
        return None  # unlabelled, not clean: every PIDray bag holds a threat (Step 1b)
    sizes = scan.sizes or [image_size(v) for v in scan.views]
    items = []
    for i in scan.items:
        name = kev_name(scan.dataset, i.label)
        items.append({
            "name": name,
            "group": group(name, scan.dataset),
            "label": i.label,
            "view": i.view,
            "box": clamp(i.box, sizes[i.view]) if i.box else None,
            "difficult": i.difficult,
        })
    meta = dict(scan.meta)
    if scan.dataset == "stcray":
        meta.update(stcray_staging(scan.views[0].stem))
    tier, licence = TIERS[scan.dataset]
    return {
        "id": str(scan.views[0].relative_to(DATA).with_suffix("")),
        "dataset": scan.dataset,
        "split": scan.split,
        "tier": tier,
        "licence": licence,
        "views": [str(v.relative_to(DATA)) for v in scan.views],
        "sizes": sizes,
        "items": items,
        "meta": meta,
    }


def build() -> list[dict]:
    lines = [line for scan in all_scans() if (line := to_line(scan)) is not None]
    ids = Counter(line["id"] for line in lines)
    duplicates = [i for i, n in ids.items() if n > 1]
    assert not duplicates, f"duplicate ids: {duplicates[:5]}"
    return lines


def read() -> list[dict]:
    with open(MANIFEST) as f:
        return [json.loads(line) for line in f]


def main() -> None:
    lines = build()
    with open(MANIFEST, "w") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"wrote {len(lines):,} scans to {MANIFEST.relative_to(DATA.parent)} ({MANIFEST.stat().st_size / 1e6:.0f} MB)")
    print(f"{'dataset':<11} {'split':<12} {'tier':<11} {'scans':>7} {'clean':>7}")
    for (dataset, split, tier), n in sorted(Counter((l["dataset"], l["split"], l["tier"]) for l in lines).items()):
        clean = sum(1 for l in lines if l["dataset"] == dataset and l["split"] == split and not l["items"])
        print(f"{dataset:<11} {split:<12} {tier:<11} {n:>7,} {clean:>7,}")
    groups = Counter((i["group"], i["name"]) for l in lines for i in l["items"] if i["view"] == 0)
    for g in ["weapon or explosive", "restricted"]:
        names = sorted(((n, name) for (gr, name), n in groups.items() if gr == g), reverse=True)
        print(f"\n{g}: {sum(n for n, _ in names):,} items across {len(names)} names")
        print("   " + ", ".join(f"{name} {n:,}" for n, name in names))


if __name__ == "__main__":
    main()
