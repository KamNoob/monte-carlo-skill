# Interval DP

Reference script: `../../dynamic-programming-workspace/reference-scripts/interval_dp.py`.

State: `dp[i][j]`, defined over a contiguous range or substring `[i, j]`, computed by trying every split point `k` inside the range. Naive complexity is O(n³): O(n²) intervals × O(n) split points each. The defining feature that separates this from two-sequence DP (`references/sequence-alignment.md`) is that both indices range over the *same* sequence, and the DP is filled by increasing interval **length**, not by increasing `i` or `j` independently — every subproblem of length `k` must be resolved before any subproblem of length `k+1` that contains it.

## Matrix-chain multiplication

Given matrices with shapes `dims[i] x dims[i+1]`, find the parenthesization minimizing total scalar multiplications:

```
dp[i][j] = min over k in [i, j-1] of:
    dp[i][k] + dp[k+1][j] + dims[i]*dims[k+1]*dims[j+1]
```

`dp[i][k]` and `dp[k+1][j]` are the costs of optimally parenthesizing the left and right sub-chains; the third term is the cost of the final multiplication combining whatever those two sub-chains produce (a `dims[i] x dims[k+1]` matrix times a `dims[k+1] x dims[j+1]` matrix). Filling by increasing chain length ensures every `dp[i][k]`/`dp[k+1][j]` used is already known.

To recover the actual parenthesization, record the minimizing `k` at each `(i,j)` in a `split` table and recursively rebuild `(A_i..A_k)(A_{k+1}..A_j)` from it — `interval_dp.py`'s `matrix_chain_parens` does this and is checked against the standard CLRS textbook instance (6 matrices, optimal cost 15125).

**Optimal Binary Search Tree** is the same recurrence shape with a different cost term (expected search cost weighted by access frequency instead of multiplication cost) — not implemented here, but a direct analogue.

## Palindrome partitioning: minimum cuts

Partition a string into the fewest pieces such that every piece is a palindrome. Two DP passes:

1. **Precompute `is_pal[i][j]`** (itself an interval DP): a substring is a palindrome iff its endpoints match and the interior (length − 2) is a palindrome — `is_pal[i][j] = (s[i]==s[j]) and (length <= 2 or is_pal[i+1][j-1])`.
2. **`cuts[j]`** = fewest cuts needed for the prefix `s[:j+1]`: 0 if the whole prefix is itself a palindrome, otherwise `min(cuts[i-1] + 1 for i where s[i:j+1] is a palindrome)`.

**Burst balloons** (LeetCode-style: burst balloons one at a time, gain = product of current balloon's value and its current neighbors', minimize/maximize total) is a less obvious interval DP: define `dp[i][j]` as the best result for bursting everything strictly *between* `i` and `j`, with `k` as the *last* balloon burst in that range rather than a splitting point — reversing the usual framing (thinking about what's burst last, not first) is the key insight that makes it fit the interval-DP mold at all. Not implemented here; worth knowing the trick exists if a problem's naive framing doesn't yield an obvious recurrence.

## Speeding up O(n³) → O(n²): when it's worth it

If the optimal split point `k` for `dp[i][j]` is provably monotone in `i` and `j` (formally, when the cost function satisfies the quadrangle inequality), **Knuth's optimization** restricts the search range for each `(i,j)` using the optimal `k` of neighboring subproblems, bringing many interval DPs from O(n³) to O(n²). This is a real speedup but requires proving monotonicity holds for your specific cost function — it isn't automatic. See `references/optimization-techniques.md`.

## When to reach for this family

The problem is about a single sequence, and the natural question is "what's the best way to combine/split/multiply/partition a contiguous range of it," where the best way to handle a range depends on trying every place to split it into two smaller ranges.
