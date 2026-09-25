from kev.metrics import auc


def test_auc():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1, 2], [3, 4]) == 0.0
    assert auc([5, 5, 5], [5, 5]) == 0.5
    assert auc([2], [2, 1]) == 0.75
    assert auc([20, 15, 30], [5, 18]) == 5 / 6
