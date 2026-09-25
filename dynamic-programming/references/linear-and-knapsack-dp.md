# 1-D Linear DP and the Knapsack Family

Reference script: `../../dynamic-programming-workspace/reference-scripts/linear_dp.py` (1-D) and `knapsack.py` (knapsack family).

## 1-D linear DP

The state `dp[i]` depends on a fixed, small window of earlier entries — usually `dp[i-1]` and/or `dp[i-2]`. This is the simplest DP shape and the one to reach for first: climbing stairs, house robber, Kadane's max subarray, coin change.

**Climbing stairs** (1 or 2 steps at a time): `ways(n) = ways(n-1) + ways(n-2)`, because the last step taken to land on `n` was either a 1-step hop from `n-1` or a 2-step hop from `n-2`. Identical recurrence to Fibonacci.

**House robber** (max sum, no two adjacent elements): `dp[i] = max(dp[i-1], dp[i-2] + values[i])` — either skip house `i` or rob it and add to the best-so-far excluding its neighbor.

**Kadane's algorithm** (max sum contiguous subarray): `best_ending_here[i] = max(values[i], best_ending_here[i-1] + values[i])` — either start a fresh subarray at `i`, or extend the running one. Track the max seen across all `i`.

All three are O(n) time, O(1) space (rolling variables) because the recurrence only ever looks back a fixed, small number of steps.

## Coin change: two problems, one recurrence shape, opposite loop nesting

**Minimum coins to make an amount** (`dp[a] = 1 + min(dp[a-c] for c in coins if c <= a)`) needs the *whole table* up to `amount`, since any coin can jump back an arbitrary distance — this is O(n · len(coins)) with the amount as the outer loop, coins as the inner loop; loop order doesn't matter here because it's a `min`, not a count.

**Number of ways to make an amount** (order-independent combinations) needs coins in the *outer* loop and amount in the *inner* loop:

```python
dp = [1] + [0] * amount
for c in coins:
    for a in range(c, amount + 1):
        dp[a] += dp[a - c]
```

Swapping the loop order (amount outer, coins inner) instead counts *permutations* — {1,2} and {2,1} would be counted as two different ways of making 3, when the problem usually wants them counted once. This loop-order distinction is a classic DP gotcha with no compiler warning to catch it; it only shows up as a wrong final count.

## 0/1 knapsack

Given item weights/values and a capacity, choose a subset maximizing value without exceeding capacity, each item usable **at most once**:

```python
dp = [0] * (capacity + 1)
for w, v in zip(weights, values):
    for c in range(capacity, w - 1, -1):   # DESCENDING
        dp[c] = max(dp[c], dp[c - w] + v)
```

The capacity loop must run **downward**. `dp[c - w]` on the right must refer to a value computed *before* the current item was folded in; iterating upward would let `dp[c-w]` already include the current item, effectively allowing it to be picked twice (turning this into the *unbounded* knapsack, where each item can be reused). This is the single most common bug in knapsack-shaped DP — it doesn't crash, it just silently answers a different problem.

O(n·capacity) time and space in the 1-D rolling form. This is **pseudo-polynomial**: exponential in the number of *bits* needed to represent the capacity, not in `n`, so it doesn't contradict knapsack being NP-hard.

### Reconstructing which items were chosen

The 1-D rolling array only gives the optimal *value*. To recover which items were taken, keep the full 2-D table `dp[i][c]` and walk backward: if `dp[i][c] != dp[i-1][c]`, item `i-1` was included and the walk continues at `dp[i-1][c - weight[i-1]]`; otherwise move to `dp[i-1][c]` unchanged. This costs O(n·capacity) space instead of O(capacity).

## Subset sum and equal-subset partition

Subset sum (can some subset hit an exact target?) is 0/1 knapsack with boolean "value" instead of numeric value — same descending-loop structure, `reachable[c] = reachable[c] or reachable[c-x]`.

Equal-subset partition (can the array be split into two subsets with equal sum?) reduces directly to subset sum with target = `total_sum / 2`: if such a subset exists, the complement automatically sums to the other half. If `total_sum` is odd, an equal split is immediately impossible — no need to run the DP at all.

## When to reach for this family vs. others

- If the recurrence only ever looks back a small fixed window → 1-D linear DP.
- If there's a resource constraint (capacity, budget, weight limit) and each item is used at most once → knapsack.
- If you need the choices, not just the value → keep the full table, not the rolled array (see reconstruction above).
