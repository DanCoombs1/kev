import re

from kev.data.text import detokenize, topic_words
from kev.questions.bank import ABSENT, ABSENT_HELD_OUT, GROUP_NOUNS_HELD_OUT, ITEMS_HELD_OUT, PRESENT, PRESENT_HELD_OUT, WHICH, WHICH_HELD_OUT


def test_detokenize_undoes_wikitext_splitting():
    assert detokenize(" red @-@ tipped , while 1 @,@ 000 of 2 @.@ 5 cm ( 24 in ) . \n") == \
        "red-tipped, while 1,000 of 2.5 cm (24 in)."
    assert detokenize(' the other is the " cutter " , which ') == 'the other is the "cutter", which'
    assert detokenize(" well , isn 't that Churchill 's ? ") == "well, isn't that Churchill's?"
    assert detokenize(" 5 – 6 kilograms ; 60 % ") == "5 – 6 kilograms; 60%"


def test_paragraphs_are_never_picked_by_held_out_wording():
    words = lambda templates: set(re.findall(r"[a-z'-]+", " ".join(templates).replace("{one}", "").replace("{many}", "")))
    held_out = words(PRESENT_HELD_OUT + ABSENT_HELD_OUT + WHICH_HELD_OUT) - words(PRESENT + ABSENT + WHICH)
    for table in (ITEMS_HELD_OUT, GROUP_NOUNS_HELD_OUT):
        for nouns in table.values():
            held_out |= {w for n in nouns for w in (n.word.lower(), n.many.lower())}
    assert "spanner" in held_out and "luggage" in held_out
    assert not topic_words() & held_out
