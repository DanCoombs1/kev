from collections import Counter, defaultdict

import pytest

from kev.data.curate import CURATED, read

if not CURATED.exists():
    pytest.skip("curated data not built: uv run python -m kev.data.curate", allow_module_level=True)

LINES = read()


def test_every_scan_has_a_role():
    assert {line["role"] for line in LINES} == {"train", "val", "test"}


def test_no_group_straddles_two_roles():
    roles = defaultdict(set)
    for line in LINES:
        roles[line["group"]].add(line["role"])
    split = {g: r for g, r in roles.items() if len(r) > 1}
    assert not split, list(split.items())[:3]


def test_official_test_sets_stay_test_and_official_training_never_becomes_test():
    for line in LINES:
        if line["split"].startswith("test"):
            assert line["role"] == "test", line["id"]
        if line["split"] == "train":
            assert line["role"] in ("train", "val"), line["id"]


def test_proportions_are_close_to_the_targets():
    for dataset in {line["dataset"] for line in LINES}:
        lines = [line for line in LINES if line["dataset"] == dataset]
        if lines[0]["split"] == "all":
            roles = Counter(line["role"] for line in lines)
            assert abs(roles["test"] / len(lines) - 0.20) < 0.04, (dataset, roles)
            assert abs(roles["val"] / len(lines) - 0.10) < 0.04, (dataset, roles)
        else:
            training = [line for line in lines if line["split"] == "train"]
            val = sum(line["role"] == "val" for line in training)
            assert abs(val / len(training) - 0.10) < 0.03, (dataset, val, len(training))


def test_every_stcray_variant_keeps_training_scans():
    roles = defaultdict(Counter)
    for line in LINES:
        if line["dataset"] == "stcray" and line["split"] == "train":
            roles[line["meta"]["variant"]][line["role"]] += 1
    starved = [v for v, r in roles.items() if r["train"] == 0]
    assert not starved, starved
