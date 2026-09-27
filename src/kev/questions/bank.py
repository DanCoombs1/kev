"""How kev's questions and options can be worded.

Held-out wording is never trained on. It's kept back to test whether kev understands wording it hasn't seen.

    uv run python -m kev.questions.bank
"""

import random
from dataclasses import dataclass

from kev.data.names import RESTRICTED, WEAPON


@dataclass(frozen=True)
class Noun:
    word: str  # as an option: "scissors"
    one: str   # "is there {one}?": "a pair of scissors"
    many: str  # "any {many}?": "scissors"


def noun(word: str, many: str | None = None, one: str | None = None) -> Noun:
    article = "an" if word[0].lower() in "aeiou" else "a"
    return Noun(word, one or f"{article} {word}", many or f"{word}s")


def pair_of(word: str) -> Noun:
    return Noun(word, f"a pair of {word}", word)


# Only true synonyms: a label of "gun" doesn't say whether it's a revolver, so "revolver" can't be here.
ITEMS = {
    "gun": [noun("gun"), noun("handgun"), noun("pistol"), noun("firearm")],
    "3D-printed gun": [noun("3D-printed gun"), noun("3D-printed pistol"), noun("printed gun")],
    "bullet": [noun("bullet"), noun("cartridge"), noun("round of ammunition", many="rounds of ammunition")],
    "explosive": [noun("explosive"), noun("explosive device"), noun("bomb"), noun("IED"),
                  noun("improvised explosive device")],
    "fireworks": [noun("firework"), noun("firecracker")],
    "modified laptop": [noun("modified laptop"), noun("tampered laptop"), noun("laptop bomb")],
    "modified mobile phone": [noun("modified mobile phone"), noun("modified phone"), noun("tampered phone")],
    "modified pager": [noun("modified pager"), noun("tampered pager")],
    "modified walkie-talkie": [noun("modified walkie-talkie"), noun("tampered walkie-talkie"), noun("modified radio")],
    "modified electronic parts": [noun("modified electronic part"), noun("tampered electronic component")],
    "knife": [noun("knife", many="knives")],
    "box cutter": [noun("box cutter"), noun("utility knife", many="utility knives", one="a utility knife"),
                   noun("retractable blade")],
    "razor blade": [noun("razor blade")],
    "razor": [noun("razor"), noun("shaving razor")],
    "saw blade": [noun("saw blade")],
    "other sharp object": [Noun("other sharp object", "another sharp object", "other sharp objects")],
    "dart": [noun("dart")],
    "baton": [noun("baton"), noun("truncheon"), noun("police baton"), noun("telescopic baton")],
    "baseball bat": [noun("baseball bat"), noun("bat")],
    "handcuffs": [pair_of("handcuffs")],
    "battery": [noun("battery", many="batteries")],
    "power bank": [noun("power bank"), noun("portable charger"), noun("battery pack"),
                   noun("external battery", many="external batteries")],
    "lighter": [noun("lighter"), noun("cigarette lighter")],
    "spray can": [noun("spray can"), noun("aerosol can"), noun("aerosol"), noun("spray bottle")],
    "syringe": [noun("syringe"), noun("hypodermic needle")],
    "scissors": [pair_of("scissors")],
    "hammer": [noun("hammer")],
    "wrench": [noun("wrench", many="wrenches")],
    "pliers": [pair_of("pliers")],
    "screwdriver": [noun("screwdriver")],
    "laptop": [noun("laptop"), noun("laptop computer")],
    "mobile phone": [noun("mobile phone"), noun("phone"), noun("cell phone"), noun("smartphone")],
    "pager": [noun("pager")],
    "walkie-talkie": [noun("walkie-talkie"), noun("handheld radio")],
    "nail clippers": [pair_of("nail clippers"), noun("nail clipper")],
}

ITEMS_HELD_OUT = {
    "gun": [noun("sidearm")],
    "3D-printed gun": [noun("ghost gun")],
    "bullet": [noun("round of ammo", many="rounds of ammo")],
    "explosive": [noun("explosive charge")],
    "fireworks": [noun("pyrotechnic device")],
    "modified laptop": [noun("rigged laptop")],
    "modified mobile phone": [noun("rigged phone")],
    "modified pager": [noun("rigged pager")],
    "modified walkie-talkie": [noun("rigged walkie-talkie")],
    "modified electronic parts": [noun("rigged electronic component")],
    "box cutter": [noun("Stanley knife", many="Stanley knives")],
    "baton": [noun("nightstick")],
    "power bank": [noun("powerbank")],
    "lighter": [noun("pocket lighter")],
    "spray can": [noun("spray canister")],
    "wrench": [noun("spanner")],
    "laptop": [noun("notebook computer")],
    "mobile phone": [noun("handset")],
    "pager": [noun("beeper")],
    "walkie-talkie": [noun("two-way radio")],
    "nail clippers": [noun("nail cutter")],
}

GROUP_NOUNS = {
    WEAPON: [noun("weapon or explosive", many="weapons or explosives"), noun("threat item"),
             Noun("dangerous item", "anything dangerous", "dangerous items")],
    RESTRICTED: [noun("restricted item"), noun("restricted object"),
                 Noun("restricted item", "anything restricted", "restricted items")],
}

GROUP_NOUNS_HELD_OUT = {
    WEAPON: [noun("threat object")],
    RESTRICTED: [noun("restricted article")],
}

# "yes" means it's there.
PRESENT = [
    "is there {one} in this bag?",
    "is there {one} in the bag?",
    "is there {one} in here?",
    "does this bag contain {one}?",
    "does the bag have {one} in it?",
    "can you see {one}?",
    "can you see {one} in this scan?",
    "do you see {one} anywhere?",
    "does the scan show {one}?",
    "is {one} visible in the scan?",
    "is {one} present?",
    "has this passenger packed {one}?",
    "is someone carrying {one}?",
    "is there {one} hidden in this bag?",
    "is this bag carrying {one}?",
    "any {many}?",
    "any {many} in this bag?",
    "are there any {many}?",
    "are there any {many} in the bag?",
    "does this bag have any {many} in it?",
    "can you spot any {many}?",
    "{many}?",
    "check for {many}",
    "look for {many} in this bag",
]

PRESENT_HELD_OUT = [
    "does this luggage have {one} inside?",
    "has anyone packed {one} in here?",
    "could there be {one} in this suitcase?",
    "do you reckon there are {many} in here?",
    "spot any {many}?",
]

# "yes" means it isn't.
ABSENT = [
    "is this bag free of {many}?",
    "is the bag clear of {many}?",
    "is this bag free from {many}?",
    "are there no {many} in this bag?",
    "can you confirm there are no {many}?",
    "is the scan free of {many}?",
    "does this bag have no {many}?",
    "is it clear of {many}?",
]

ABSENT_HELD_OUT = [
    "no {many} in here, right?",
    "is it safe to say there are no {many}?",
]

# The options are items, plus "none of these".
WHICH = [
    "which of these is in the bag?",
    "which of these is in this bag?",
    "which of these items can you see?",
    "which one of these is in the scan?",
    "which of the following is present?",
    "which of these has this passenger packed?",
    "what is in this bag?",
    "what item is in the bag?",
    "what can you see in this bag?",
    "which item does this bag contain?",
    "what's in here?",
    "name the item in this bag",
]

WHICH_HELD_OUT = [
    "which of these is inside the luggage?",
    "what has been packed in this suitcase?",
    "pick the item you can see",
]

# The options are items, and any number of them can be right.
WHICH_ALL = [
    "which of these are in the bag?",
    "which of these are in this bag?",
    "which of these items are present?",
    "which of the following can you see? pick all that apply",
    "select everything that is in this bag",
    "what is in this bag? choose all that apply",
    "tick every item you can see in the scan",
    "which of these does the bag contain?",
    "mark all of these that are in the bag",
    "which of these has this passenger packed? pick all that apply",
]

WHICH_ALL_HELD_OUT = [
    "which of these are inside the luggage?",
    "pick every item packed in this suitcase",
]

YES, NO = "yes", "no"


def ask(template: str, thing: Noun) -> str:
    return template.format(one=thing.one, many=thing.many)


def _nouns(*tables: dict[str, list[Noun]]) -> list[Noun]:
    return [n for table in tables for nouns in table.values() for n in nouns]


def training_text() -> list[str]:
    """Every training question and option, for the tokenizer."""
    things = _nouns(ITEMS, GROUP_NOUNS)
    questions = [ask(t, n) for t in PRESENT + ABSENT for n in things]
    return questions + WHICH + WHICH_ALL + [n.word for n in _nouns(ITEMS)] + [YES, NO]


def held_out_questions() -> dict[str, list[str]]:
    old_things, new_things = _nouns(ITEMS, GROUP_NOUNS), _nouns(ITEMS_HELD_OUT, GROUP_NOUNS_HELD_OUT)
    old_templates, new_templates = PRESENT + ABSENT, PRESENT_HELD_OUT + ABSENT_HELD_OUT
    return {
        "new word": [ask(t, n) for t in old_templates for n in new_things],
        "new sentence": [ask(t, n) for t in new_templates for n in old_things] + WHICH_HELD_OUT + WHICH_ALL_HELD_OUT,
        "both new": [ask(t, n) for t in new_templates for n in new_things],
    }


def main() -> None:
    rng = random.Random(0)
    train = training_text()
    print(f"items: {len(ITEMS)}, {len(_nouns(ITEMS))} training names, {len(_nouns(ITEMS_HELD_OUT))} held out")
    print(f"groups: {len(GROUP_NOUNS)}, {len(_nouns(GROUP_NOUNS))} training names, {len(_nouns(GROUP_NOUNS_HELD_OUT))} held out")
    print(f"sentences: present {len(PRESENT)} (+{len(PRESENT_HELD_OUT)} held out), "
          f"absent {len(ABSENT)} (+{len(ABSENT_HELD_OUT)}), which {len(WHICH)} (+{len(WHICH_HELD_OUT)}), "
          f"which-all {len(WHICH_ALL)} (+{len(WHICH_ALL_HELD_OUT)})")
    print(f"training text: {len(set(train)):,} distinct lines, {len(' '.join(train).split()):,} words\n")
    for line in rng.sample(train, 12):
        print(f"  {line}")
    for kind, questions in held_out_questions().items():
        print(f"\nheld out, {kind}: {len(set(questions)):,}")
        for line in rng.sample(questions, 4):
            print(f"  {line}")


if __name__ == "__main__":
    main()
