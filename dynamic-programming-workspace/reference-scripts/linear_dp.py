"""1-D DP: problems where dp[i] depends on a fixed-size window of earlier dp[i-k].

All functions are O(n) time, O(1) space (rolling variables), except
coin_change_min and coin_change_ways which are O(n*len(coins)) with an
O(n) table (the recurrence needs the whole table, not just a window).
"""
from __future__ import annotations


def climbing_stairs(n: int) -> int:
    """Number of ways to climb n stairs, taking 1 or 2 steps at a time.

    ways(n) = ways(n-1) + ways(n-2): the last step taken to reach stair n
    was either a 1-step hop from stair n-1, or a 2-step hop from n-2.
    """
    if n <= 1:
        return 1
    a, b = 1, 1  # ways(0), ways(1)
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def house_robber(values: list[float]) -> float:
    """Max sum of a subset of `values` with no two adjacent elements chosen.

    dp[i] = max(dp[i-1], dp[i-2] + values[i])
      -> either skip house i (keep dp[i-1]), or rob it (dp[i-2] + values[i]).
    """
    prev2 = prev1 = 0.0
    for v in values:
        prev2, prev1 = prev1, max(prev1, prev2 + v)
    return prev1


def max_subarray(values: list[float]) -> float:
    """Kadane's algorithm: max sum of a contiguous subarray.

    best_ending_here[i] = max(values[i], best_ending_here[i-1] + values[i])
      -> either start a fresh subarray at i, or extend the previous best.
    """
    if not values:
        return 0.0
    best_ending_here = best_overall = values[0]
    for v in values[1:]:
        best_ending_here = max(v, best_ending_here + v)
        best_overall = max(best_overall, best_ending_here)
    return best_overall


def coin_change_min(coins: list[int], amount: int) -> int:
    """Fewest coins to make `amount`, or -1 if impossible.

    dp[a] = 1 + min(dp[a-c] for c in coins if c <= a), dp[0] = 0.
    Unbounded knapsack shape: each coin may be reused, so the amount loop
    runs upward (contrast with 0/1 knapsack in knapsack.py, which must
    run downward to avoid reusing an item).
    """
    INF = float("inf")
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount] if dp[amount] != INF else -1


def coin_change_ways(coins: list[int], amount: int) -> int:
    """Number of distinct combinations (order-independent) making `amount`.

    Iterating coins in the OUTER loop and amount in the inner loop counts
    combinations, not permutations: each coin denomination is "decided"
    once across the whole amount range before moving to the next coin, so
    {1,2} and {2,1} are the same combination and counted once.
    """
    dp = [1] + [0] * amount
    for c in coins:
        for a in range(c, amount + 1):
            dp[a] += dp[a - c]
    return dp[amount]


if __name__ == "__main__":
    assert climbing_stairs(5) == 8
    assert climbing_stairs(0) == 1
    assert house_robber([2, 7, 9, 3, 1]) == 12  # rob houses 0,2,4: 2+9+1
    assert house_robber([]) == 0
    assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6  # [4,-1,2,1]
    assert coin_change_min([1, 2, 5], 11) == 3  # 5+5+1
    assert coin_change_min([2], 3) == -1
    assert coin_change_ways([1, 2, 5], 5) == 4  # 5; 2+2+1; 2+1+1+1; 1*5
    print("linear_dp: all checks passed")
