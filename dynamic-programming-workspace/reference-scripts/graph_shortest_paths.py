"""Shortest-path DP over a graph's edges.

These are DP because the state (shortest distance to a node) is built
from the states of its predecessors, and the same sub-distances are
reused across many nodes' relaxations.

- Bellman-Ford: handles negative edge weights, detects negative cycles,
  O(V*E). dp here is "shortest path using at most k edges", relaxed V-1
  times (any shortest simple path has at most V-1 edges).
- Floyd-Warshall: all-pairs shortest paths, O(V^3). dp[k][i][j] = shortest
  path from i to j using only intermediate nodes 0..k-1; rolls the k
  dimension in place since dp[k] only reads dp[k-1].

If all weights are non-negative, Dijkstra's algorithm (greedy + a heap,
not DP) is faster for single-source: O(E log V).
"""
from __future__ import annotations
import math


def bellman_ford(n: int, edges: list[tuple[int, int, float]], source: int):
    """edges: list of (u, v, weight) for directed edge u->v.

    Returns (dist, has_negative_cycle). dist[v] is math.inf if unreachable.
    Relaxing V-1 times settles every shortest simple path; if a V-th pass
    still relaxes some edge, a negative-weight cycle is reachable from
    source and shortest paths through it are undefined (unbounded below).
    """
    dist = [math.inf] * n
    dist[source] = 0.0
    for _ in range(n - 1):
        changed = False
        for u, v, w in edges:
            if dist[u] != math.inf and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:  # early exit: nothing relaxed, already converged
            break
    has_negative_cycle = False
    for u, v, w in edges:
        if dist[u] != math.inf and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break
    return dist, has_negative_cycle


def floyd_warshall(n: int, edges: list[tuple[int, int, float]]):
    """All-pairs shortest paths. Returns an n x n distance matrix.

    dp[i][j] considering intermediates 0..k-1 vs. 0..k:
        dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j])
    i.e. "does routing through k improve the current best i->j distance?"
    The k-loop must be outermost: dp[i][k] and dp[k][j] on the right must
    already reflect intermediates up to k-1, which only holds if k has
    been fully processed as the outer loop variable before i, j vary.
    """
    dist = [[math.inf] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0.0
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)  # keep the cheapest of parallel edges
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik == math.inf:
                continue
            row = dist[i]
            for j in range(n):
                cand = dik + dk[j]
                if cand < row[j]:
                    row[j] = cand
    return dist


if __name__ == "__main__":
    # Bellman-Ford: classic CLRS example (5 nodes, one negative edge, no negative cycle)
    edges = [(0, 1, 6), (0, 3, 7), (1, 2, 5), (1, 3, 8), (1, 4, -4),
              (2, 1, -2), (3, 2, -3), (3, 4, 9), (4, 0, 2), (4, 2, 7)]
    dist, has_neg = bellman_ford(5, edges, source=0)
    assert not has_neg
    assert dist == [0, 2, 4, 7, -2], dist

    # Negative cycle detection
    cyc_edges = [(0, 1, 1), (1, 2, -3), (2, 0, 1)]  # cycle sums to -1
    _, has_neg = bellman_ford(3, cyc_edges, source=0)
    assert has_neg

    # Floyd-Warshall cross-check against Bellman-Ford single-source column
    fw = floyd_warshall(5, edges)
    assert fw[0] == dist, (fw[0], dist)

    # Unreachable node
    dist2, _ = bellman_ford(3, [(0, 1, 1)], source=0)
    assert dist2[2] == math.inf
    print("graph_shortest_paths: all checks passed")
