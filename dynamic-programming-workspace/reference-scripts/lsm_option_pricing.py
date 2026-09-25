"""Where dynamic programming meets Monte Carlo: American option pricing.

An American/Bermudan option's holder chooses, at each exercise date,
between exercising now and continuing:

    V_t = max(payoff_t, E[discount * V_{t+1} | S_t])

This is a finite-horizon DP (backward induction from expiry) exactly like
`mdp_value_iteration.py`'s value function, except the "action" is
binary (exercise / continue) and the state is the asset price.

- On a binomial/trinomial lattice, the conditional expectation is exact
  at each node, so backward induction (`american_put_binomial` below) is
  itself the whole algorithm: no sampling needed, no bias to worry about.
- With Monte Carlo *simulated paths*, that conditional expectation is
  unknown. Longstaff & Schwartz (2001, Review of Financial Studies
  14:113-147) estimate it by regressing realized discounted continuation
  values onto basis functions of the current price, using only paths
  that are in-the-money (`american_put_lsm` below). This is approximate
  dynamic programming: the exact Bellman backup is replaced by a
  regression fit to Monte Carlo samples, the same substitution
  `references/stochastic-dp-mdp.md` describes generically.

LSM is a biased-low estimator for two independent reasons: (1) it prices
a Bermudan option with a finite number of exercise dates rather than
continuous exercise, and (2) the regressed exercise policy is only an
approximation to the optimal one, and following a suboptimal policy can
only do as well as or worse than optimal. For a clean lower bound,
estimate the policy on one set of paths and price on an independent set;
for an upper bound, use Andersen-Broadie duality (not implemented here).
"""
from __future__ import annotations
import numpy as np


def american_put_binomial(S0: float, K: float, r: float, sigma: float, T: float,
                           steps: int = 2000) -> float:
    """Cox-Ross-Rubinstein binomial lattice, exact backward induction.

    At each node, V = max(exercise payoff, discounted expected continuation
    under the risk-neutral probability p). No Monte Carlo noise; converges
    to the true continuous-exercise price as steps -> infinity.
    """
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    S = S0 * u ** np.arange(steps, -1, -1) * d ** np.arange(0, steps + 1)
    V = np.maximum(K - S, 0)
    for i in range(steps - 1, -1, -1):
        S = S0 * u ** np.arange(i, -1, -1) * d ** np.arange(0, i + 1)
        continuation = disc * (p * V[:-1] + (1 - p) * V[1:])
        V = np.maximum(K - S, continuation)
    return V[0]


def american_put_lsm(S0: float, K: float, r: float, sigma: float, T: float,
                      steps: int = 50, n_paths: int = 100_000, seed: int = 42,
                      degree: int = 2) -> float:
    """Longstaff-Schwartz least-squares Monte Carlo.

    Simulates GBM paths, then works backward from expiry: at each date,
    regress the discounted cash flow each in-the-money path actually
    received later against a polynomial in the current price (the basis
    functions), and exercise now wherever the immediate payoff beats that
    fitted continuation value.
    """
    rng = np.random.default_rng(seed)
    dt = T / steps
    half = n_paths // 2
    Z = rng.standard_normal((half, steps))
    Z = np.vstack([Z, -Z])  # antithetic variates: halves variance for free
    logS = np.log(S0) + np.cumsum(
        (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z, axis=1
    )
    S = np.exp(logS)
    cash = np.maximum(K - S[:, -1], 0.0)  # value if held to expiry

    for t in range(steps - 2, -1, -1):
        cash *= np.exp(-r * dt)  # discount one step back
        itm = K - S[:, t] > 0  # regress on in-the-money paths only (Longstaff-Schwartz's own restriction, reduces regression noise from far-OTM paths)
        if not itm.any():
            continue
        x = S[itm, t] / K
        X = np.column_stack([x**k for k in range(degree + 1)])
        beta, *_ = np.linalg.lstsq(X, cash[itm], rcond=None)
        continuation = X @ beta
        exercise = K - S[itm, t]
        ex_now = exercise > continuation
        idx = np.where(itm)[0][ex_now]
        cash[idx] = exercise[ex_now]

    return np.exp(-r * dt) * cash.mean()


if __name__ == "__main__":
    # Longstaff & Schwartz (2001) Table 1 test case: S0=36, K=40, r=6%, sigma=20%, T=1
    b2000 = american_put_binomial(36, 40, 0.06, 0.2, 1.0, steps=2000)
    b4000 = american_put_binomial(36, 40, 0.06, 0.2, 1.0, steps=4000)
    assert abs(b2000 - b4000) < 1e-3, (b2000, b4000)  # lattice has converged

    lsm = american_put_lsm(36, 40, 0.06, 0.2, 1.0)
    assert lsm < b2000, (lsm, b2000)          # expected low bias (see module docstring)
    assert abs(lsm - b2000) < 0.05, (lsm, b2000)  # but not wildly off

    print(f"binomial (converged) = {b2000:.4f}")
    print(f"LSM (50 dates, 100k paths) = {lsm:.4f}")
    print("lsm_option_pricing: all checks passed")
