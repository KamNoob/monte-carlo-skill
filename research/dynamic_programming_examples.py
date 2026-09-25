"""Runnable, self-checking versions of the examples in dynamic-programming.md.

Run: python3 dynamic_programming_examples.py   (needs numpy)
"""
from functools import lru_cache
import numpy as np


# 1. Top-down memoization vs bottom-up tabulation
@lru_cache(maxsize=None)
def fib_memo(n):
    return n if n < 2 else fib_memo(n - 1) + fib_memo(n - 2)


def fib_tab(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# 2. 0/1 knapsack, O(nW) time, O(W) space
def knapsack(weights, values, W):
    dp = [0] * (W + 1)
    for w, v in zip(weights, values):
        for c in range(W, w - 1, -1):  # descending: each item used at most once
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[W]


# 3. Edit distance (Levenshtein)
def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = a[i - 1] != b[j - 1]
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]


# 4. Longest increasing subsequence, O(n log n) patience variant
def lis_length(xs):
    import bisect
    tails = []
    for x in xs:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


# 5. Value iteration on a tiny MDP: P[a][s, s'], R[s, a]
def value_iteration(P, R, gamma=0.9, tol=1e-10):
    V = np.zeros(R.shape[0])
    while True:
        Q = R + gamma * np.stack([P[a] @ V for a in range(len(P))], axis=1)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, Q.argmax(axis=1)
        V = V_new


# 6. American put: binomial backward induction (exact DP on a lattice)
def american_put_binomial(S0, K, r, sigma, T, steps=2000):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt)); d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    S = S0 * u ** np.arange(steps, -1, -1) * d ** np.arange(0, steps + 1)
    V = np.maximum(K - S, 0)
    for i in range(steps - 1, -1, -1):
        S = S0 * u ** np.arange(i, -1, -1) * d ** np.arange(0, i + 1)
        V = np.maximum(K - S, disc * (p * V[:-1] + (1 - p) * V[1:]))
    return V[0]


# 7. American put: Longstaff-Schwartz least-squares Monte Carlo (approximate DP)
def american_put_lsm(S0, K, r, sigma, T, steps=50, n_paths=100_000, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / steps
    half = n_paths // 2
    Z = rng.standard_normal((half, steps))
    Z = np.vstack([Z, -Z])                      # antithetic variates
    logS = np.log(S0) + np.cumsum((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1)
    S = np.exp(logS)
    cash = np.maximum(K - S[:, -1], 0)          # value if held to expiry
    for t in range(steps - 2, -1, -1):
        cash *= np.exp(-r * dt)                 # discount one step back
        itm = K - S[:, t] > 0                   # regress on in-the-money paths only
        if itm.sum() == 0:
            continue
        x = S[itm, t] / K
        X = np.column_stack([np.ones_like(x), x, x**2])
        beta, *_ = np.linalg.lstsq(X, cash[itm], rcond=None)
        continuation = X @ beta
        exercise = K - S[itm, t]
        ex_now = exercise > continuation
        idx = np.where(itm)[0][ex_now]
        cash[idx] = exercise[ex_now]
    return np.exp(-r * dt) * cash.mean()


if __name__ == "__main__":
    assert fib_memo(90) == fib_tab(90) == 2880067194370816120
    assert knapsack([1, 3, 4, 5], [1, 4, 5, 7], 7) == 9
    assert edit_distance("kitten", "sitting") == 3
    assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4

    # 2-state MDP: action 0 = stay, action 1 = switch. Reward only for being in state 1.
    P = [np.eye(2), np.array([[0, 1], [1, 0]], float)]
    R = np.array([[0.0, 0.0], [1.0, 1.0]])
    V, pi = value_iteration(P, R, gamma=0.9)
    assert np.allclose(V, [9.0, 10.0]), V   # state 0: switch then earn 1/(1-0.9) discounted
    assert list(pi) == [1, 0]

    # Longstaff-Schwartz (2001) Table 1 case: S0=36, K=40, r=6%, sigma=20%, T=1
    b = american_put_binomial(36, 40, 0.06, 0.2, 1.0, steps=2000)
    b2 = american_put_binomial(36, 40, 0.06, 0.2, 1.0, steps=4000)
    m = american_put_lsm(36, 40, 0.06, 0.2, 1.0)
    print(f"binomial(2000)={b:.4f}  binomial(4000)={b2:.4f}  lsm={m:.4f}")
    assert abs(b - b2) < 1e-3   # lattice has converged
    assert m < b                # 50 exercise dates (Bermudan) + suboptimal policy -> low bias
    assert abs(m - b) < 0.05  # LSM is a slightly low-biased estimator with 50 exercise dates
    print("all DP examples verified")
