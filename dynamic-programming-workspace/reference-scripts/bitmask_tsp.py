"""Bitmask DP: dp[mask][i] where mask is a subset of items/cities as a bitmask.

Held-Karp for the Travelling Salesman Problem: O(2^n * n^2) time,
O(2^n * n) space. Exact and exponential — practical up to roughly n=20
on ordinary hardware, after which approximation/heuristic methods
(nearest neighbor + 2-opt, Lin-Kernighan, branch and bound) take over.
This is the standard example of bitmask DP; the same "state = which
subset have I used/visited, plus where am I now" pattern applies to
other subset-covering and assignment problems.
"""
from __future__ import annotations
import math


def tsp_held_karp(dist: list[list[float]]) -> tuple[float, list[int]]:
    """Shortest cycle visiting every city exactly once, returning to the start.

    dp[mask][i] = min cost of a path that starts at city 0, visits exactly
    the cities in `mask` (which always includes 0 and i), and ends at city i.

    dp[mask][i] = min over j in mask, j != i of:
        dp[mask without i][j] + dist[j][i]

    Base case: dp[{0}][0] = 0 (starting point, no travel yet).
    Answer: min over i != 0 of dp[full_mask][i] + dist[i][0] (close the loop).
    """
    n = len(dist)
    if n == 1:
        return 0.0, [0]
    full_mask = (1 << n) - 1
    dp = [[math.inf] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0.0  # mask={0}, at city 0

    for mask in range(1 << n):
        if not (mask & 1):  # city 0 must always be in the visited set
            continue
        for i in range(n):
            if not (mask & (1 << i)) or dp[mask][i] == math.inf:
                continue
            for j in range(n):
                if mask & (1 << j):
                    continue  # j already visited
                new_mask = mask | (1 << j)
                cand = dp[mask][i] + dist[i][j]
                if cand < dp[new_mask][j]:
                    dp[new_mask][j] = cand
                    parent[new_mask][j] = i

    best_cost = math.inf
    best_last = -1
    for i in range(1, n):
        cand = dp[full_mask][i] + dist[i][0]
        if cand < best_cost:
            best_cost, best_last = cand, i

    # Reconstruct the path by walking parent pointers backward.
    path = []
    mask, i = full_mask, best_last
    while i != -1:
        path.append(i)
        prev = parent[mask][i]
        mask ^= (1 << i)
        i = prev
    path.reverse()
    return best_cost, path


def tour_length(dist: list[list[float]], path: list[int]) -> float:
    return sum(dist[path[k]][path[(k + 1) % len(path)]] for k in range(len(path)))


if __name__ == "__main__":
    # 4-city square with side 1: optimal tour goes around the perimeter, cost 4.
    dist = [
        [0, 1, 2, 1],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [1, 2, 1, 0],
    ]
    cost, path = tsp_held_karp(dist)
    assert cost == 4, cost
    assert sorted(path) == [0, 1, 2, 3]
    assert tour_length(dist, path) == cost

    # Brute force cross-check on a random 8-city instance.
    import itertools
    import random

    rng = random.Random(0)
    n = 8
    pts = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(n)]
    d = [[math.dist(pts[a], pts[b]) for b in range(n)] for a in range(n)]
    brute_best = min(
        tour_length(d, [0, *perm])
        for perm in itertools.permutations(range(1, n))
    )
    hk_cost, hk_path = tsp_held_karp(d)
    assert abs(hk_cost - brute_best) < 1e-9, (hk_cost, brute_best)
    print("bitmask_tsp: all checks passed")
