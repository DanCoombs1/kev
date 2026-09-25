"""Scores for judging kev and the checks around it."""


def auc(positives: list[float], negatives: list[float]) -> float:
    """Chance a random positive scores higher than a random negative. 0.5 means the score says nothing.

    Ties count as half a win: tied values share the average of their ranks.
    """
    values = sorted(positives + negatives)
    rank = {}
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and values[j] == values[i]:
            j += 1
        rank[values[i]] = (i + 1 + j) / 2
        i = j
    rank_sum = sum(rank[v] for v in positives)
    return (rank_sum - len(positives) * (len(positives) + 1) / 2) / (len(positives) * len(negatives))
