# Bitmask DP and Graph Shortest-Path DP

Reference scripts: `../../dynamic-programming-workspace/reference-scripts/bitmask_tsp.py` and `graph_shortest_paths.py`.

## Bitmask DP: state includes "which subset have I used so far"

When a problem needs to track an unordered subset of a small universe (n ≲ 20) as part of the state, represent the subset as an integer bitmask and let it be one axis of the DP table. `dp[mask][i]` = best value achievable having used exactly the items/cities in `mask`, currently "at" item `i`. Complexity is O(2ⁿ · n) states with O(n) work per transition = **O(2ⁿ · n²)** — exponential in `n`, but for `n ≤ ~20` this is around 20 million states, which is tractable, versus `n!` for brute-force permutation search (20! is astronomically larger).

### Held-Karp for the Travelling Salesman Problem

Shortest cycle visiting every city exactly once and returning to the start:

```
dp[mask][i] = min cost of a path starting at city 0, visiting exactly the
              cities in mask (which always contains 0 and i), ending at i

dp[mask][i] = min over j in mask, j != i:
    dp[mask without i][j] + dist[j][i]
```

Base case `dp[{0}][0] = 0`. Final answer: `min over i != 0 of dp[full_mask][i] + dist[i][0]` (close the loop back to the start). Reconstructing the actual tour needs a parent pointer per `(mask, i)` state, walked backward from the best final `i`.

This is the standard example of bitmask DP, but the *pattern* — "state = (which subset has been handled, where am I now)" — generalizes to other subset-covering and assignment problems (e.g. assigning n workers to n tasks with a specific-subset-so-far state). Past `n ≈ 20`, exact DP becomes infeasible and the field shifts to heuristics (nearest-neighbor + 2-opt, Lin-Kernighan) or branch-and-bound with pruning, which are no longer DP in the tabulation sense.

## Graph shortest-path DP

Both algorithms below are DP because a node's shortest-path state is built from its predecessors' states, and the same partial distances are reused across many nodes' relaxations — not because they're usually taught under a "DP" heading.

### Bellman-Ford: single source, handles negative weights

```
Relax every edge (u, v, w): if dist[u] + w < dist[v], update dist[v].
Repeat this for all edges, V-1 times total.
```

The state here is implicitly "shortest path using at most `k` edges," and any shortest *simple* path has at most `V-1` edges, so `V-1` full passes are guaranteed to have converged. **A negative-weight cycle reachable from the source makes "shortest path" undefined** (you could loop the cycle forever, driving the cost to −∞) — detect this by running one more relaxation pass after the `V-1`: if anything still improves, a negative cycle exists and downstream distances through it are meaningless. O(V·E).

### Floyd-Warshall: all pairs, O(V³)

```
dp[k][i][j] = shortest i -> j path using only intermediate nodes {0, ..., k-1}
dp[k][i][j] = min(dp[k-1][i][j], dp[k-1][i][k] + dp[k-1][k][j])
```

i.e., "does routing through node `k` beat the current best known `i -> j` distance using only nodes before `k`?" The `k` dimension can be rolled in place (2-D array updated in a specific loop order: `k` outermost, then `i`, then `j`) since `dp[k]` only ever reads `dp[k-1]`. `graph_shortest_paths.py` cross-checks a single row of Floyd-Warshall's output against Bellman-Ford run from the same source — they must agree since both compute the true shortest-path distance.

### When to use which

- **Non-negative weights, single source** → Dijkstra's algorithm (greedy + a min-heap, not DP) — O(E log V), faster than Bellman-Ford.
- **Possible negative weights, single source, need to detect negative cycles** → Bellman-Ford.
- **All-pairs distances needed, V is small-to-medium** (roughly V ≤ ~500 for O(V³) to be practical) → Floyd-Warshall.
- **All-pairs, V large, weights non-negative** → run Dijkstra from every source instead (O(V·E log V)), usually cheaper than O(V³) when the graph is sparse.

## When to reach for this family

Bitmask DP: a small (≤ ~20) set of items/locations that must all be used/visited, where the order or exact subset used-so-far matters to future decisions. Graph shortest-path DP: the problem is literally about shortest paths in a weighted graph, especially with negative weights (ruling out Dijkstra) or a need for all-pairs distances at once.
