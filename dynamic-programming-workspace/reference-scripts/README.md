# Dynamic Programming Reference Scripts

Importable, self-checking Python implementations backing the [`dynamic-programming` skill](../../dynamic-programming/README.md). Each module is documented in more depth in that skill's `references/*.md` files — this README is just the setup + index.

## Setup

Everything here is pure Python except `mdp_value_iteration.py` and `lsm_option_pricing.py`, which need `numpy`. If you already have the venv from `monte-carlo-workspace/reference-scripts/`, reuse it — don't create a second one:

```bash
python3 -m venv .venv        # only if you don't already have one to reuse
.venv/bin/pip install numpy
```

Run any script directly — each one's `if __name__ == "__main__":` block is a self-check against known-correct results (a textbook example, a closed-form solution, or a brute-force cross-check), not just a demo:

```bash
.venv/bin/python3 knapsack.py
# knapsack: all checks passed
```

Or import from your own code:

```python
import sys
sys.path.insert(0, "/path/to/reference-scripts")
from sequence_alignment import edit_distance
```

## Modules

- **`linear_dp.py`** — 1-D DP with O(1) rolling state: climbing stairs, house robber, Kadane's max subarray, coin change (min coins and count of ways — the two use different loop orders, which is itself the point being illustrated).
- **`knapsack.py`** — 0/1 knapsack (with and without item reconstruction), subset sum, equal-subset partition. All O(n·capacity); the capacity loop direction (descending) is what makes it 0/1 rather than unbounded, and is regression-tested against reuse explicitly.
- **`sequence_alignment.py`** — Longest Common Subsequence (length and reconstruction) and edit distance (Levenshtein), including an O(min(m,n))-space edit distance variant, cross-checked against the full 2-D version.
- **`interval_dp.py`** — Matrix-chain multiplication (optimal parenthesization, reconstructed) and minimum-cut palindrome partitioning, both classic O(n³) interval DPs.
- **`bitmask_tsp.py`** — Held-Karp exact Travelling Salesman Problem via bitmask DP, O(2ⁿ·n²); cross-checked against brute-force permutation search on a random instance.
- **`graph_shortest_paths.py`** — Bellman-Ford (handles negative weights, detects negative cycles) and Floyd-Warshall (all-pairs), cross-checked against each other on the same graph.
- **`mdp_value_iteration.py`** — Value iteration and policy iteration for finite MDPs (the stochastic-DP / Bellman-equation case), cross-checked against each other and against a closed-form solution on a toy 2-state MDP.
- **`lsm_option_pricing.py`** — Where DP meets this repo's other skill: Longstaff-Schwartz least-squares Monte Carlo for American option pricing (approximate DP — a regression fit replaces the exact Bellman backup), validated against a converged binomial-lattice backward induction (exact DP, no sampling).

## What each self-check actually validates

Every module's `__main__` block asserts against something independently verifiable — not just "it ran": textbook examples with known answers (CLRS's matrix-chain and Bellman-Ford instances), brute-force cross-checks (TSP against permutation search, edit distance's linear-space variant against the full table), closed-form solutions (the toy MDP's value function), or cross-validation between two different algorithms that must agree because they solve the same problem (value iteration vs. policy iteration, Bellman-Ford vs. Floyd-Warshall, the binomial lattice as ground truth for LSM's expected low bias). One bug was caught this way during development: an initial subset-sum test target turned out to actually be reachable (3+5+2=10), not unreachable as assumed — the fix was to verify the "impossible" test case by brute force before trusting it, rather than trusting arithmetic done by eye.

## License

MIT — see `../../dynamic-programming/LICENSE`.
