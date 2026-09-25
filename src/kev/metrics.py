def auc(positives: list[float], negatives: list[float]) -> float:
    """Probability that a random positive scores above a random negative, counting ties as half."""
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
