"""Builds data/manifest.jsonl: every scan from every dataset in one format, with kev item names.

    uv run python -m kev.data.manifest
"""

import json
import re
from collections import Counter

from kev.data.names import group, kev_name
from kev.data.sources import DATA, Scan, all_scans, image_size

MANIFEST = DATA / "manifest.jsonl"

TIERS = {
    "stcray": ("commercial", "CC BY 4.0 (paper) / Apache-2.0 (Hugging Face)"),
    "iedxray": ("commercial", "CC BY 4.0"),
    "compass_xp": ("commercial", "CC BY 4.0"),
    "pidray": ("academic", "academic use only"),
    "dvxray": ("academic", "no licence stated; treated as academic use only"),
}


def stcray_staging(stem: str) -> dict:
    """Parse STCray's filename code, e.g. Gun4_B1_L10_C10_Loc1_phi1_th1_1.

    Timestamp-named test scans have no code. A few explosive scans use a different scheme,
    which is kept as-is under "other".
    """
    if re.match(r"\d{4}-\d{2}-\d{2}", stem):
        return {}
    parts = stem.split("_")
    staging = {"variant": parts[0]}
    for part in parts[1:]:
        if part.isdigit():
            staging["shot"] = part
        elif m := re.fullmatch(r"(B|L|C|Loc|phi|th)(\d+[A-Z]?)", part):
            staging[m.group(1)] = m.group(2)
        else:
            staging["other"] = part
    return staging


def clamp(box: tuple[float, float, float, float], size: tuple[int, int]) -> list[float]:
    x, y, w, h = box
    width, height = size
    x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
    return [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)]


def to_line(scan: Scan) -> dict | None:
    if scan.dataset == "pidray" and not scan.items:
        return None  # every PIDray bag holds a threat, so these are unlabelled rather than clean
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
    duplicates = [i for i, n in Counter(line["id"] for line in lines).items() if n > 1]
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
