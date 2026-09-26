from kev.data.text import detokenize


def test_detokenize_undoes_wikitext_splitting():
    assert detokenize(" red @-@ tipped , while 1 @,@ 000 of 2 @.@ 5 cm ( 24 in ) . \n") == \
        "red-tipped, while 1,000 of 2.5 cm (24 in)."
    assert detokenize(' the other is the " cutter " , which ') == 'the other is the "cutter", which'
    assert detokenize(" well , isn 't that Churchill 's ? ") == "well, isn't that Churchill's?"
    assert detokenize(" 5 – 6 kilograms ; 60 % ") == "5 – 6 kilograms; 60%"
