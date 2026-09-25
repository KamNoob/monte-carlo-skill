# Speeding Up a Correct DP

Apply these only after the plain DP is correct and verified — they're optimizations to a working solution, not shortcuts to one. None are implemented as standalone reference scripts in this skill (they're targeted techniques applied on top of the problem families in the other references); this doc explains when each applies and where the payoff comes from.

## Rolling arrays

If `dp[i][...]` only ever reads from `dp[i-1][...]` (never `dp[i-2]` or earlier), keep just the previous row instead of the full table. Turns O(m·n) space into O(n). `sequence_alignment.py`'s `edit_distance_linear_space` does exactly this. **The tradeoff:** you lose the ability to reconstruct the actual sequence of choices (traceback needs the full table, or the technique below).

## Hirschberg's algorithm: linear-space reconstruction

When you need both O(m+n) space *and* the actual alignment/path (not just the optimal value), rolling arrays alone aren't enough. Hirschberg's algorithm finds the midpoint of the optimal alignment using two linear-space passes (forward from the start, backward from the end, meeting in the middle row), then recurses on each half. Same O(mn) total time as the full table, small constant-factor overhead from the two passes, but O(m+n) space instead of O(mn) — the difference between feasible and not for very long sequences.

## Monotone queue / deque optimization

When a DP transition is a sliding-window `min` or `max` (e.g. `dp[i] = values[i] + min(dp[i-k..i-1])`), a naive scan over the window is O(k) per state, O(nk) total. A monotone deque maintains the window's min/max incrementally in amortized O(1) per state, bringing the total to O(n).

## Convex hull trick / Li Chao tree

When a DP transition has the form `dp[i] = min_j (m_j · x_i + b_j)` — each earlier state `j` defines a line, and you want the lowest line at a given `x_i` — a naive scan is O(n) per query, O(n²) total. Maintaining the *lower envelope* of the lines (via a monotonic stack when insertion order is nice, or a Li Chao tree in the general case) answers each query in O(log n), bringing the total to O(n log n).

## Divide-and-conquer optimization / Knuth's optimization

Both apply when the optimal split point of a DP transition is **monotone** in the surrounding indices. Knuth's optimization (a specific case, applicable when the cost function satisfies the quadrangle inequality) restricts each subproblem's search range using the optimal split point of its neighbors, turning many O(n³) interval DPs (see `references/interval-dp.md`) into O(n²) — this is not automatic and requires checking the monotonicity condition holds for the specific cost function in question. Divide-and-conquer optimization is the more general form, applicable to a 1-D DP layer whose optimal predecessor index is monotone across states, cutting an O(n²) layer to O(n log n).

## Meet-in-the-middle

For an exponential state space that's too large to enumerate directly (e.g. subset-sum over 40 items, where 2⁴⁰ is too many), split the input in half, enumerate all `2^(n/2)` possibilities for each half independently, then combine the two halves efficiently (typically by sorting one half and binary-searching against it, or via a hash join). Turns O(2ⁿ) into O(2^(n/2) · n), often the difference between infeasible and a few seconds.

## Matrix exponentiation

Any DP whose recurrence is a fixed **linear** combination of a fixed number of previous states (e.g. `dp[n] = a·dp[n-1] + b·dp[n-2]`) can be expressed as a matrix-vector recurrence `v_n = M · v_{n-1}`, and then `v_n = M^n · v_0`. Computing `M^n` by repeated squaring takes O(k³ log n) instead of O(n) — turns Fibonacci's O(n) tabulation into O(log n), which matters when `n` is astronomically large (e.g. `n ~ 10^18`) rather than merely large.

## When it's *not* worth it

If the plain DP already runs fast enough for the actual input sizes involved, none of the above is worth the added complexity and bug surface. Verify correctness first (see the Step 4 validation checklist in `SKILL.md`), then optimize only if profiling or the problem's stated constraints (e.g. `n ≤ 10^5` with a 1-second time limit, ruling out O(n²)) actually demand it.
