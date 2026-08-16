def weighted_vote(neighbors):

    weights = {}

    for distance, label in neighbors:

        weight = 1 / (distance + 0.000001)

        weights[label] = \
            weights.get(label, 0) + weight

    return max(
        weights,
        key=weights.get
    )


# Example
neighbors = [
    (1.0, 0),
    (2.0, 1),
    (3.0, 1)
]

result = weighted_vote(neighbors)

print("Weighted prediction:", result)