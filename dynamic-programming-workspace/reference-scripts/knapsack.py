"""0/1 knapsack and its direct relatives (subset sum, equal-subset partition).

Every function here is O(n*W) time and O(W) space, where W is the
capacity/target — pseudo-polynomial: exponential in the *bit length* of W,
not in n. Fine for W up to a few million; use meet-in-the-middle or a
different approach for astronomically large W.
"""
from __future__ import annotations


def knapsack_01(weights: list[int], values: list[float], capacity: int) -> float:
    """Max value from items of given weight/value, each usable at most once.

    dp[c] = best value achievable with total weight <= c, using items
    considered so far. Iterating capacity DOWNWARD when folding in each
    item is what makes it 0/1 (each item's contribution to dp[c] can only
    come from a dp[c-w] computed *before* this item was added). Iterating
    upward instead would allow the same item to be reused arbitrarily
    many times (that's the *unbounded* knapsack, a different problem).
    """
    dp = [0.0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[capacity]


def knapsack_01_with_items(
    weights: list[int], values: list[float], capacity: int
) -> tuple[float, list[int]]:
    """Same as knapsack_01, but also reconstructs which item indices were taken.

    Needs the full 2-D table (can't use the rolling 1-D trick and still
    recover the choices), so this is O(n*W) time AND space.
    """
    n = len(weights)
    dp = [[0.0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]
            if w <= c:
                dp[i][c] = max(dp[i][c], dp[i - 1][c - w] + v)
    # Walk backward: at each row, did including item i change the value?
    chosen = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            chosen.append(i - 1)
            c -= weights[i - 1]
    chosen.reverse()
    return dp[n][capacity], chosen


def subset_sum_possible(nums: list[int], target: int) -> bool:
    """Can some subset of nums sum to exactly target? (0/1 knapsack, boolean values.)"""
    reachable = [False] * (target + 1)
    reachable[0] = True
    for x in nums:
        for c in range(target, x - 1, -1):
            if reachable[c - x]:
                reachable[c] = True
    return reachable[target]


def can_partition_equal_subset(nums: list[int]) -> bool:
    """Can nums be split into two subsets with equal sum?

    Equivalent to subset_sum_possible(nums, total/2): if such a subset
    exists, the rest automatically sums to the other half.
    """
    total = sum(nums)
    if total % 2 != 0:
        return False
    return subset_sum_possible(nums, total // 2)


if __name__ == "__main__":
    assert knapsack_01([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9  # items of weight 3+4
    best, items = knapsack_01_with_items([1, 3, 4, 5], [1, 4, 5, 7], 7)
    assert best == 9
    assert sum(v for i, v in enumerate([1, 4, 5, 7]) if i in items) == 9
    assert sum(w for i, w in enumerate([1, 3, 4, 5]) if i in items) <= 7

    assert subset_sum_possible([3, 34, 4, 12, 5, 2], 9) is True   # 4+5
    assert subset_sum_possible([3, 34, 4, 12, 5, 2], 59) is False  # verified unreachable by brute force
    assert can_partition_equal_subset([1, 5, 11, 5]) is True   # {1,5,5} vs {11}
    assert can_partition_equal_subset([1, 2, 3, 5]) is False

    # unbounded-vs-0/1 regression guard: with reuse allowed, weight-3 item
    # alone could fill capacity 6 for value 8; 0/1 must not do that with
    # only one copy of each item.
    assert knapsack_01([3], [4], 6) == 4  # can't take the item twice
    print("knapsack: all checks passed")
