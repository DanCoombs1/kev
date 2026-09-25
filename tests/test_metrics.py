from kev.metrics import auc


def test_auc_on_cases_with_known_answers():
    assert auc([3, 4], [1, 2]) == 1.0  # every positive above every negative
    assert auc([1, 2], [3, 4]) == 0.0  # perfectly backwards is still a perfect separator
    assert auc([5, 5, 5], [5, 5]) == 0.5  # all tied: says nothing
    assert auc([2], [2, 1]) == 0.75  # one win, one tie (half a win), out of two pairs
    assert auc([20, 15, 30], [5, 18]) == 5 / 6  # the worked example from the AUC explanation
