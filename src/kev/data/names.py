"""Maps each dataset's labels to kev item names and screening groups."""

WEAPON = "weapon or explosive"
RESTRICTED = "restricted"
ALLOWED = "allowed"

NAMES = {
    ("pidray", "Baton"): "baton",
    ("pidray", "Bullet"): "bullet",
    ("pidray", "Gun"): "gun",
    ("pidray", "Hammer"): "hammer",
    ("pidray", "HandCuffs"): "handcuffs",
    ("pidray", "Knife"): "knife",
    ("pidray", "Lighter"): "lighter",
    ("pidray", "Pliers"): "pliers",
    ("pidray", "Powerbank"): "power bank",
    ("pidray", "Scissors"): "scissors",
    ("pidray", "Sprayer"): "spray can",
    ("pidray", "Wrench"): "wrench",
    ("stcray", "3D printed gun"): "3D-printed gun",
    ("stcray", "Battery"): "battery",
    ("stcray", "Blade"): "razor blade",
    ("stcray", "Bullet"): "bullet",
    ("stcray", "Cutter"): "box cutter",
    ("stcray", "Explosive"): "explosive",
    ("stcray", "Gun"): "gun",
    ("stcray", "Hammer"): "hammer",
    ("stcray", "Handcuffs"): "handcuffs",
    ("stcray", "Knife"): "knife",
    ("stcray", "Lighter"): "lighter",
    ("stcray", "Nail Cutter"): "nail clippers",
    ("stcray", "Other Sharp Item"): "other sharp object",
    ("stcray", "Pliers"): "pliers",
    ("stcray", "Powerbank"): "power bank",
    ("stcray", "Scissors"): "scissors",
    ("stcray", "Screwdriver"): "screwdriver",
    ("stcray", "Shaving Razor"): "razor",
    ("stcray", "Syringe"): "syringe",
    ("stcray", "Wrench"): "wrench",
    ("dvxray", "Bat"): "baseball bat",
    ("dvxray", "Battery"): "battery",
    ("dvxray", "Dart"): "dart",
    ("dvxray", "Fireworks"): "fireworks",
    ("dvxray", "Gun"): "gun",
    ("dvxray", "Hammer"): "hammer",
    ("dvxray", "Knife"): "knife",
    ("dvxray", "Lighter"): "lighter",
    ("dvxray", "Pliers"): "pliers",
    ("dvxray", "Pressure_vessel"): "spray can",
    ("dvxray", "Razor_blade"): "razor blade",
    ("dvxray", "Saw_blade"): "saw blade",
    ("dvxray", "Scissors"): "scissors",
    ("dvxray", "Screwdriver"): "screwdriver",
    ("dvxray", "Wrench"): "wrench",
    ("iedxray", "Battery"): "battery",
    ("iedxray", "Explosive"): "explosive",
    ("iedxray", "Laptop"): "laptop",
    ("iedxray", "Mobile Phone"): "mobile phone",
    ("iedxray", "Modified Mobile phone"): "modified mobile phone",
    ("iedxray", "Modified Pager"): "modified pager",
    ("iedxray", "Modified Walkie Talkie"): "modified walkie-talkie",
    ("iedxray", "Modified laptop"): "modified laptop",
    ("iedxray", "Modified parts"): "modified electronic parts",
    ("iedxray", "Pager"): "pager",
    ("iedxray", "Walkie-Talkie"): "walkie-talkie",
}

COMPASS_RENAMES = {"razor_blades": "razor blade"}

GROUPS = {
    **dict.fromkeys([
        "gun", "3D-printed gun", "bullet", "explosive", "fireworks",
        "modified laptop", "modified mobile phone", "modified pager", "modified walkie-talkie", "modified electronic parts",
        "knife", "bread knife", "carving knife", "craft knife", "penknife", "dagger",
        "box cutter", "razor blade", "razor", "saw blade", "other sharp object", "letter opener", "hatchet",
        "dart", "baton", "baseball bat", "handcuffs",
    ], WEAPON),
    **dict.fromkeys([
        "battery", "power bank", "lighter", "blowtorch", "spray can", "gas canister", "syringe",
        "scissors", "nail scissors", "hammer", "mallet", "wrench", "pliers", "screwdriver",
        "chisel", "crowbar", "saw", "hacksaw", "pickaxe", "plane", "power drill", "secateurs", "staple gun",
        "wirecutter", "laser pointer", "screw", "nail", "hook", "corkscrew",
    ], RESTRICTED),
    **dict.fromkeys(["laptop", "mobile phone", "pager", "walkie-talkie", "nail clippers"], ALLOWED),
}

KIND_OF = {
    "bread knife": "knife", "carving knife": "knife", "craft knife": "knife", "penknife": "knife", "dagger": "knife",
    "nail scissors": "scissors", "mallet": "hammer",
}


def kev_name(dataset: str, label: str) -> str:
    if dataset == "compass_xp":
        return COMPASS_RENAMES.get(label, label.replace("_", " "))
    return NAMES[(dataset, label)]


def group(name: str, dataset: str) -> str:
    if name in GROUPS:
        return GROUPS[name]
    if dataset == "compass_xp":  # everything COMPASS-XP flags as harmless
        return ALLOWED
    raise KeyError(f"no group for {name!r} from {dataset}")
