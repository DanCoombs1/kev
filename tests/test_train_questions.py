import torch

from kev.model.kev import ANY_OF, ONE_OF, YES_NO
from kev.questions.generate import Question
from kev.train.questions import correct, nothing_there, presence, question_loss


class Stub:
    any_of_bias = torch.tensor(0.0)

    def probabilities(self, scores, q):
        from kev.model.kev import Kev
        return Kev.probabilities(self, scores, q)


def batch() -> dict:
    """A yes/no ("yes" is right), a one-of (the third option is right) and an any-of (first and third right, second unscored)."""
    q = {"type": torch.tensor([[YES_NO, ONE_OF, ANY_OF]]), "valid": torch.tensor([[True, True, True]]),
         "option_valid": torch.tensor([[[True, True, False], [True, True, True], [True, True, True]]]),
         "answer": torch.tensor([[[True, False, False], [False, False, True], [True, False, True]]]),
         "scored": torch.tensor([[[True, True, False], [True, True, True], [True, False, True]]])}
    return q


def test_loss_is_small_when_confidently_right_and_large_when_wrong():
    q = batch()
    right = torch.tensor([[[9.0, -9, 0], [-9, -9, 9], [9, 0, 9]]])
    wrong = -right
    assert question_loss(Stub(), right, q) < 0.01 < 5 < question_loss(Stub(), wrong, q)


def test_unscored_options_dont_count():
    q = batch()
    right = torch.tensor([[[9.0, -9, 0], [-9, -9, 9], [9, 0, 9]]])
    moved = right.clone()
    moved[0, 2, 1] = 50  # the unscored any-of option
    assert torch.isclose(question_loss(Stub(), right, q), question_loss(Stub(), moved, q))
    assert correct(Stub(), moved, q).tolist() == [[True, True, True]]


def test_empty_bag_guess():
    q = lambda t, answer, options, negated=False: Question(t, "", options, answer, [True] * len(options), [None] * len(options), negated)
    assert nothing_there(q(YES_NO, [False, True], ["yes", "no"]))
    assert not nothing_there(q(YES_NO, [False, True], ["yes", "no"], negated=True))
    assert nothing_there(q(ONE_OF, [False, True], ["knife", None]))
    assert not nothing_there(q(ANY_OF, [True, False], ["knife", "gun"]))


def test_presence_reads_through_negation_and_skips_unscored_options():
    yes_no = Question(YES_NO, "", ["yes", "no"], [True, False], [True, True], ["knife"], negated=True)
    assert presence(yes_no, torch.tensor([0.9, 0.1])) == [(False, False)]  # "free of knives?" yes: not there, says so
    any_of = Question(ANY_OF, "", ["knife", "gun", "bat"], [True, False, False], [True, False, True], ["knife", "gun", "bat"])
    assert presence(any_of, torch.tensor([0.8, 0.9, 0.7])) == [(True, True), (False, True)]
