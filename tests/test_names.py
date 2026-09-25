from collections import defaultdict

from kev.data.names import ALLOWED, GROUPS, KIND_OF, group, kev_name
from kev.data.sources import all_scans

LABELS = defaultdict(set)
DANGEROUS = {}
for scan in all_scans():
    for item in scan.items:
        LABELS[scan.dataset].add(item.label)
        if scan.dataset == "compass_xp":
            DANGEROUS[item.label] = scan.meta["dangerous"]


def test_every_label_has_a_name_and_group():
    for dataset, labels in LABELS.items():
        for label in labels:
            group(kev_name(dataset, label), dataset)


def test_compass_dangerous_flag_agrees_with_groups():
    # We restrict all tools and syringes; COMPASS-XP marks its pliers, wrenches and syringes harmless.
    stricter = {"pliers", "wrench", "syringe"}
    for label, dangerous in DANGEROUS.items():
        name = kev_name("compass_xp", label)
        our_group = group(name, "compass_xp")
        if name in stricter:
            assert our_group != ALLOWED and not dangerous
            continue
        assert (our_group != ALLOWED) == dangerous, f"{label}: dangerous={dangerous}, group={our_group}"


def test_general_names_exist():
    for specific, general in KIND_OF.items():
        assert specific in GROUPS and general in GROUPS


def test_names_are_plain_lower_case():
    for name in GROUPS:
        assert name == name.lower() or name.startswith("3D"), name
        assert "_" not in name, name
