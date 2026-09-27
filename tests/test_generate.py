import random

from kev.data.names import GROUPS, RESTRICTED, WEAPON
from kev.model.kev import ANY_OF, ONE_OF, YES_NO
from kev.questions.bank import ABSENT, ITEMS, PRESENT, WHICH, WHICH_ALL
from kev.questions.generate import ASKABLE, QUESTIONS_PER_SCAN, question_batch, questions_for
from kev.tokenizer.bpe import Tokenizer, train, count_chunks
from kev.questions.bank import training_text


def bag(dataset: str, *names: str, pair: bool = False, yes_only: bool = False) -> dict:
    return {"dataset": dataset, "items": [{"name": n, "group": GROUPS[n]} for n in names], "pair": pair, "yes_only": yes_only}


def many(sample: dict, n: int = 300) -> list:
    rng = random.Random(0)
    return [q for _ in range(n) for q in questions_for(sample, rng)]


def test_yes_no_answers_follow_the_bag_and_the_wording():
    for q in many(bag("pidray", "knife", "lighter")):
        if q.type == YES_NO:
            there = q.about[0] in ("knife", "lighter")
            assert q.answer == [there != q.negated, there == q.negated]
            templates = ABSENT if q.negated else PRESENT
            assert any(q.text == t.format(one=n.one, many=n.many) for t in templates for n in ITEMS[q.about[0]])


def test_a_kind_of_counts_as_yes():
    answers = {q.answer[0] for q in many(bag("stcray", "3D-printed gun")) if q.type == YES_NO and q.about == ["gun"]}
    assert answers == {True, False}  # yes, or no when worded the other way round
    for q in many(bag("stcray", "3D-printed gun")):
        if q.type == YES_NO and q.about == ["gun"]:
            assert q.answer[0] != q.negated


def test_ambiguous_items_are_never_answered():
    for q in many(bag("stcray", "box cutter")):
        if q.type in (YES_NO, ONE_OF):
            assert "knife" not in q.about and "other sharp object" not in q.about
        if q.type == ANY_OF and "knife" in q.about:
            assert not q.scored[q.about.index("knife")]


def test_one_of_has_exactly_one_right_answer():
    for q in many(bag("pidray", "knife", "gun")):
        if q.type == ONE_OF:
            assert sum(q.answer) == 1 and q.text in WHICH and None in q.options
            assert q.answer[q.options.index(None)] == (not {"knife", "gun"} & set(q.about))
        if q.type == ANY_OF:
            assert q.text in WHICH_ALL and q.answer == [a in ("knife", "gun") for a in q.about]


def test_only_labelled_items_are_asked_about_at_real_life_rates():
    questions = many(bag("pidray", "knife"), n=2000)
    asked = [q.about[0] for q in questions if q.type == YES_NO]
    assert set(asked) <= set(ASKABLE["pidray"])
    assert abs(asked.count("knife") / len(asked) - 1 / len(ASKABLE["pidray"])) < 0.02


def test_who_gets_which_questions():
    assert many(bag("pidray", "knife", yes_only=True), 5) == []
    assert many(bag("dvxray"), 5) == []
    assert len(many(bag("iedxray", "laptop"), 1)) == QUESTIONS_PER_SCAN
    clean, inserted = questions_for(bag("stcray", pair=True), random.Random(1)), questions_for(bag("stcray", "lighter", pair=True), random.Random(1))
    assert [q.about for q in clean[:2]] == [[WEAPON], [RESTRICTED]]
    assert [q.answer[0] != q.negated for q in clean[:2]] == [False, False]
    assert [q.answer[0] != q.negated for q in inserted[:2]] == [False, True]  # a lighter is restricted, not a weapon
    assert all(q.about[0] not in (WEAPON, RESTRICTED) for q in many(bag("stcray", "knife"), 20))


def test_batch_marks_questions_options_and_none():
    tokenizer = Tokenizer(train(count_chunks(training_text()), 100))
    rng = random.Random(0)
    per_scan = [questions_for(bag("pidray", "knife"), rng), questions_for(bag("stcray", pair=True), rng)[:3]]
    b = question_batch(per_scan, tokenizer)
    assert b["valid"].sum(dim=1).tolist() == [len(per_scan[0]), 3]
    for s, qs in enumerate(per_scan):
        for n, q in enumerate(qs):
            assert b["option_valid"][s, n].sum() == len(q.options)
            assert b["is_none"][s, n, : len(q.options)].tolist() == [o is None for o in q.options]
            assert b["answer"][s, n, : len(q.options)].tolist() == q.answer
            assert tokenizer.decode(b["ids"][s, n][b["mask"][s, n]].tolist()) == q.text.lower()
