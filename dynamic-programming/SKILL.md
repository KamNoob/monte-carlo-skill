---
name: dynamic-programming
description: |
  Full dynamic programming (DP) skill — use whenever a problem asks for an optimal value (min/max), a count of ways, or a feasibility check, and a brute-force recursion would re-solve the same subproblem many times. Triggers on: "dynamic programming", "memoize", "tabulation", "knapsack", "edit distance", "longest common subsequence", "longest increasing subsequence", "coin change", "matrix chain multiplication", "optimal BST", "travelling salesman" / "TSP", "shortest path" (Bellman-Ford/Floyd-Warshall), "value iteration", "Markov decision process" / "MDP", "optimal stopping", "American option pricing", "Bellman equation", "overlapping subproblems", "optimal substructure", or "how many ways to...". Even if the user doesn't name DP explicitly — if a recursive brute force on the problem would repeat the same arguments, or the problem breaks into a sequence of decisions where each decision's best choice depends on a compact summary of the past, this skill applies. Do not use for problems where subproblems don't overlap (plain divide-and-conquer, e.g. merge sort) or where a greedy choice is provably optimal (e.g. Huffman coding, activity selection) — DP still gives the right answer there but is unnecessary machinery.
---

# Dynamic Programming Skill

Dynamic programming solves a problem by breaking it into overlapping subproblems, solving each one **once**, storing the result, and building the final answer from stored results. The two things that must both hold before DP applies:

| Property | Meaning | If missing |
|---|---|---|
| **Optimal substructure** | An optimal solution contains optimal solutions to its subproblems (Bellman's Principle of Optimality) | DP gives wrong answers (e.g. longest *simple* path in a general graph) |
| **Overlapping subproblems** | The same subproblems recur many times | DP is correct but no faster than divide-and-conquer |

## Step 1: Identify the Problem Type

Before writing any code, classify the problem — it determines the state and the reference to use:

| Type | Signal phrases | Approach | Reference |
|------|---------------|----------|-----------|
| **1-D sequential** | "climbing stairs", "house robber", "max subarray", "coin change", "how many ways to reach..." | `dp[i]` from a fixed window of earlier `dp[i-k]` | `references/linear-and-knapsack-dp.md` |
| **Knapsack / subset** | "0/1 knapsack", "subset sum", "partition into equal subsets", capacity + item constraint | `dp[i][capacity]`, capacity loop direction matters (0/1 vs unbounded) | `references/linear-and-knapsack-dp.md` |
| **Two-sequence alignment** | "longest common subsequence", "edit distance", "sequence alignment", "diff two strings" | `dp[i][j]` over two prefixes | `references/sequence-alignment.md` |
| **Interval / range** | "matrix chain multiplication", "optimal BST", "palindrome partitioning", "burst balloons" | `dp[i][j]` over a range, split at some `k` | `references/interval-dp.md` |
| **Subset-as-bitmask** | "travelling salesman" / "TSP", "assignment problem", small n (≤ ~20) over "visit every..." | `dp[mask][i]`, O(2ⁿ·n²) | `references/bitmask-and-graph-dp.md` |
| **Graph shortest paths** | "shortest path with negative weights", "all-pairs shortest path", "detect negative cycle" | Bellman-Ford (O(VE)) or Floyd-Warshall (O(V³)) | `references/bitmask-and-graph-dp.md` |
| **Sequential decisions under uncertainty** | "Markov decision process" / "MDP", "optimal stopping", "value iteration", "American/Bermudan option", "inventory control policy" | Bellman equation, value/policy iteration, or backward induction | `references/stochastic-dp-mdp.md` |
| **Plain DP too slow** | Correct DP recurrence, but O(n²) or O(n³) times out | Rolling arrays, monotone queue, convex hull trick, Knuth/D&C optimization, matrix exponentiation | `references/optimization-techniques.md` |

Ask the user one clarifying question if the type is genuinely ambiguous (most often: "do you need the optimal value, or also the actual sequence of choices?" — the latter needs a full table for reconstruction, not just a rolling window). If it's clear, proceed.

## Step 2: Design the DP

Walk through this before coding — this is where almost all the difficulty lives:

1. **Define the state.** The smallest set of parameters that determines the answer to a subproblem. Too little and the recurrence is wrong (even if it happens to pass small tests); too much and you waste time/space.
2. **Write the recurrence.** Express `dp[state]` in terms of strictly smaller states, as a `min`/`max`/`sum` over the available choices.
3. **Base cases.** The trivial states (empty prefix, zero capacity, single node, terminal time).
4. **Evaluation order.** Every state a recurrence reads must already be computed — a topological order of the subproblem dependency graph. (Bottom-up: get this right explicitly. Top-down: recursion + memoization gets it right automatically.)
5. **Answer location + reconstruction.** Which state is the final answer. If you also need the actual choices (not just the optimal value), keep back-pointers or an argmax table — you generally can't reconstruct from a space-optimized rolling array alone.
6. **Complexity.** Roughly (# states) × (work per transition). Then look for space savings: if a row only depends on the previous row, keep 1–2 rows instead of the full table.

## Step 3: Implement

### Top-down vs. bottom-up

```python
from functools import lru_cache

@lru_cache(maxsize=None)     # top-down: only computes states actually reached
def fib_memo(n):
    return n if n < 2 else fib_memo(n - 1) + fib_memo(n - 2)

def fib_tab(n):               # bottom-up: O(1) space, no recursion limit
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```

Use top-down when the reachable state space is much smaller than the full space, or the recurrence is easiest to write recursively. Use bottom-up for tight loops, when you need to control memory precisely (rolling arrays), or when Python's recursion limit (~1000) would bite.

### Reference scripts

`../dynamic-programming-workspace/reference-scripts/` has a runnable, self-checking module per problem family listed in Step 1's table (`linear_dp.py`, `knapsack.py`, `sequence_alignment.py`, `interval_dp.py`, `bitmask_tsp.py`, `graph_shortest_paths.py`, `mdp_value_iteration.py`, `lsm_option_pricing.py`). Each one's `__main__` block validates against a textbook example, a closed-form answer, or a brute-force cross-check — read the relevant one before writing new code, and adapt it rather than starting from scratch. Needs only the standard library except `mdp_value_iteration.py` and `lsm_option_pricing.py`, which need `numpy` (reuse the venv at `../monte-carlo-workspace/reference-scripts/.venv/` if this is installed alongside that skill).

## Step 4: Validate

DP bugs are almost always silent — the code runs, returns a plausible-looking number, and is wrong. Before trusting a result:

1. **Check base cases by hand** for the smallest 2–3 inputs.
2. **Cross-check against a brute-force version** on small inputs (exponential recursion without memoization, or literal enumeration) — this is what every reference script here does.
3. **Check the loop direction** on any knapsack-shaped DP: descending capacity = 0/1 (each item once), ascending = unbounded (items reusable). This is the single most common DP bug.
4. **If reconstructing a solution** (not just the optimal value), verify the reconstructed choices actually produce the claimed optimal value when replayed.
5. **For stochastic DP / MDPs**, cross-check value iteration against policy iteration (see `mdp_value_iteration.py`) — they solve the same Bellman equation and must agree.

## Common pitfalls

- **Wrong or incomplete state** — if the true answer depends on something not captured in the state, the recurrence is wrong regardless of how careful the rest of the code is.
- **Off-by-one** between `dp[i]` (a prefix *length*) and the corresponding element index `a[i-1]`.
- **Unhashable arguments** to `lru_cache` — pass tuples, not lists.
- **Pseudo-polynomial ≠ polynomial** — O(n·W) knapsack is exponential in the bit-length of `W`; the problem is still NP-hard.
- **Forgetting reconstruction** — computing the optimal *value* doesn't give you the optimal *sequence of choices* unless you kept back-pointers along the way.

See `references/pitfalls-and-checklist.md` for the full list, and how DP relates to divide-and-conquer, greedy, and backtracking.
