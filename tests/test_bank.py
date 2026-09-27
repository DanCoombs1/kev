import re

from kev.data.names import NAMES
from kev.questions.bank import (
    ABSENT, ABSENT_HELD_OUT, GROUP_NOUNS, GROUP_NOUNS_HELD_OUT, ITEMS, ITEMS_HELD_OUT, PRESENT, PRESENT_HELD_OUT,
    WHICH, WHICH_ALL, WHICH_ALL_HELD_OUT, WHICH_HELD_OUT, held_out_questions, training_text,
)


def test_every_item_we_train_on_has_wording():
    assert set(NAMES.values()) <= set(ITEMS)
    assert set(ITEMS_HELD_OUT) <= set(ITEMS)


def test_no_word_names_two_things():
    owner = {}
    for table in (ITEMS, ITEMS_HELD_OUT, GROUP_NOUNS, GROUP_NOUNS_HELD_OUT):
        for thing, nouns in table.items():
            for n in nouns:
                assert owner.setdefault(n.word.lower(), thing) == thing, n.word


def test_held_out_wording_never_reaches_training():
    text = "\n".join(training_text()).lower()
    for table in (ITEMS_HELD_OUT, GROUP_NOUNS_HELD_OUT):
        for nouns in table.values():
            for n in nouns:
                assert not re.search(rf"\b{re.escape(n.word.lower())}\b", text), n.word
    assert not set(PRESENT_HELD_OUT + ABSENT_HELD_OUT + WHICH_HELD_OUT + WHICH_ALL_HELD_OUT) & set(PRESENT + ABSENT + WHICH + WHICH_ALL)


def test_sentences_have_the_right_slots():
    for template in PRESENT + ABSENT + PRESENT_HELD_OUT + ABSENT_HELD_OUT:
        assert ("{one}" in template) != ("{many}" in template), template
    for template in WHICH + WHICH_HELD_OUT + WHICH_ALL + WHICH_ALL_HELD_OUT:
        assert "{" not in template, template


def test_rendered_questions_are_clean():
    lines = training_text() + [q for qs in held_out_questions().values() for q in qs]
    for line in lines:
        assert "{" not in line and "  " not in line and line == line.strip(), line
        assert not re.search(r"\ba [aeiou]", line.lower()) or "a utility" in line, line
        assert not re.search(r"\ban [^aeiou\s]", line.lower()), line
