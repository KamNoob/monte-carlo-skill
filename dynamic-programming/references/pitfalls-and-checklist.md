# Common Pitfalls, DP vs. Related Paradigms, and a Quick Checklist

## Common pitfalls

- **Wrong or incomplete state.** If the true answer depends on something not captured in the state, the recurrence is wrong — and this can pass small hand-checked cases while still being wrong in general, because the missing information happens not to matter on those particular inputs.
- **Wrong loop direction in knapsack-shaped DP.** Descending capacity = 0/1 (item used at most once); ascending = unbounded (item reusable). See `references/linear-and-knapsack-dp.md` — this is the single most common DP bug and produces a plausible, wrong number rather than a crash.
- **Wrong loop nesting for "count the ways."** Coin-in-outer-loop vs. amount-in-outer-loop changes whether you're counting combinations or permutations. See `references/linear-and-knapsack-dp.md`.
- **Off-by-one between prefix length and element index.** `dp[i]` usually represents a prefix of length `i`, so the corresponding array element is `a[i-1]`, not `a[i]`. Easy to get backward at a base case or a loop boundary.
- **Unhashable arguments to `functools.lru_cache`.** Pass tuples, not lists, to a memoized function; a `TypeError` here is at least loud, but converting the input silently (e.g. `tuple(x)` every call) can mask a bug where two logically-different lists happen to produce the same tuple.
- **Recursion depth in top-down Python.** The default recursion limit (~1000) is easy to hit on top-down DP over a large input. Either convert to bottom-up, or restructure the recursion to be iterative with an explicit stack.
- **Pseudo-polynomial is not polynomial.** O(n·W) knapsack is exponential in the *bit length* of `W` (doubling `W` roughly doubles the work, exactly what "exponential in bits" means) — the problem remains NP-hard, and this DP is only practical because `W` in the specific instance is small in absolute value, not because the DP secretly beat NP-hardness.
- **Forgetting reconstruction.** Computing the optimal *value* is a different (usually easier) problem than recovering the optimal *sequence of choices* that achieves it. If you need the choices, keep back-pointers or an argmax table along the way — you generally can't recover them after the fact from a space-optimized rolling array.
- **Not validating a stochastic DP's Bellman equation the same way you'd validate a deterministic one.** Value iteration converging doesn't by itself prove the model (transition matrix, reward, discount) was set up correctly — cross-check against policy iteration or a closed-form answer where one exists (see `references/stochastic-dp-mdp.md`).

## DP vs. related paradigms

| Paradigm | Subproblems | How the choice is made |
|---|---|---|
| Divide & conquer | Independent (no overlap) | Solve each side once, combine |
| **Dynamic programming** | Overlapping | Try all choices at each state, keep the best, **remember it** |
| Greedy | One subproblem per step | Commit to a single locally-best choice and never revisit it |
| Backtracking / branch & bound | Search tree | Explore, prune infeasible or provably-dominated branches |

**Greedy is DP's tempting shortcut** — it's simpler and faster (usually O(n log n) or O(n)) but is only *correct* when the problem has the **greedy-choice property**: a locally optimal choice is guaranteed to be part of some globally optimal solution. This has to be proven, not assumed. Classic contrast: the coin-change *minimum-coins* problem is greedy-solvable for the US coin system (1, 5, 10, 25) — always take the largest coin that fits — but greedy **fails** for a coin system like {1, 3, 4} making 6: greedy takes 4 + 1 + 1 (3 coins), while the true optimum is 3 + 3 (2 coins). Whenever it's unclear whether the greedy-choice property holds, default to DP — it's always correct when optimal substructure holds, whether or not the stronger greedy property also happens to hold.

**Divide and conquer degenerates to DP's cousin when subproblems overlap.** Merge sort's subproblems (disjoint halves of the array) never overlap, so there's nothing to memoize and no DP benefit. The moment recursive subproblems start repeating — as in naive recursive Fibonacci, where `fib(n-2)` is recomputed inside both the `fib(n-1)` and `fib(n-2)` branches — divide and conquer becomes exponentially wasteful, and adding memoization is precisely what turns it into DP.

## Quick checklist: is this a DP problem?

1. Does the problem ask for an optimal value (min/max), a count of ways, or a feasibility check (yes/no)?
2. Would a brute-force recursive solution call itself with the *same arguments* more than once? (If subproblems never repeat, this is plain divide-and-conquer, not DP — still fine to solve recursively, just no memoization benefit.)
3. Do the choices at each step depend only on a **compact summary of the past** (the Markov property for the state you've chosen), rather than on the full history of choices made so far? That summary is your state.
4. Is the resulting state space small enough to enumerate (directly, or via bitmask/hashing tricks)? If yes, tabulate it. If the state space is too large or continuous — a common outcome in stochastic/sequential-decision problems — see `references/stochastic-dp-mdp.md` for approximate DP, and consider whether Monte Carlo sampling (this repository's `monte-carlo` skill) is the right way to approximate the expectations involved.
