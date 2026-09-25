# Stochastic DP: MDPs, the Bellman Equation, and the Link to Monte Carlo

Reference script: `../../dynamic-programming-workspace/reference-scripts/mdp_value_iteration.py` (exact tabular case), `lsm_option_pricing.py` (the Monte Carlo / approximate DP case).

Everything else in this skill is deterministic: the recurrence's inputs fully determine its output. This reference covers DP where the state transitions randomly, and the value function itself has to account for that randomness — the point where dynamic programming and Monte Carlo simulation meet.

## The Bellman optimality equation

For a Markov Decision Process with states `s`, actions `a`, transition probabilities `P(s'|s,a)`, immediate reward `R(s,a)`, and discount factor `γ < 1`:

```
V*(s) = max_a [ R(s,a) + γ · Σ_s' P(s'|s,a) · V*(s') ]
```

This is the stochastic generalization of every deterministic recurrence elsewhere in this skill: instead of transitioning to one deterministic next state, an action leads to a *distribution* over next states, and the DP has to take an expectation over that distribution.

### Finite horizon: backward induction

If there's a known terminal time `T` (an optimal-stopping problem, a finite-horizon inventory policy, an option's expiry), `V_T` is known outright (e.g. the terminal payoff), and `V_{T-1}, V_{T-2}, ..., V_0` are computed by applying the Bellman equation moving backward in time. No iteration to convergence is needed — each `V_t` is computed exactly once, in order.

### Infinite horizon: value iteration and policy iteration

Without a terminal time, apply the Bellman operator repeatedly until it stops changing:

```python
def value_iteration(P, R, gamma=0.9, tol=1e-10):
    V = np.zeros(R.shape[0])
    while True:
        Q = R + gamma * np.stack([P[a] @ V for a in range(len(P))], axis=1)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, Q.argmax(axis=1)
        V = V_new
```

The Bellman operator is a `γ`-contraction in sup-norm (Banach fixed-point theorem), so this converges geometrically to the unique optimal `V*` regardless of the starting guess. **Policy iteration** is an alternative: evaluate the current policy exactly (solve a linear system, `mdp_value_iteration.py`'s `policy_evaluation`), improve it greedily, repeat. It typically needs far fewer iterations, at higher per-iteration cost (an `S × S` linear solve instead of one matrix-vector product). `mdp_value_iteration.py` cross-checks both against each other and against a closed-form answer on a toy 2-state MDP — they must converge to the same `V*` since both solve the same Bellman equation.

## The curse of dimensionality — and how Monte Carlo answers it

Bellman coined this phrase for exactly this setting: the state space size grows exponentially with the number of state variables, so exact tabular DP (a table entry per state) becomes infeasible once the state has more than a handful of dimensions. The response is **approximate dynamic programming (ADP)**, which reinforcement learning is largely built on: replace the exact table with a function approximator, and replace the exact expectation `Σ_s' P(s'|s,a) V(s')` — which requires knowing the full transition model — with **Monte Carlo samples** of it.

| Method | What replaces the exact Bellman backup |
|---|---|
| Monte Carlo policy evaluation | `V(s) ≈` average of sampled returns from `s` |
| TD learning / Q-learning | A single-sample stochastic backup: `Q ← Q + α(r + γ·max Q' − Q)` |
| Fitted value iteration / fitted Q-iteration | Regression of many sampled backups onto a set of features |
| **Longstaff-Schwartz (LSM)** | Regression of realized discounted continuation values onto basis functions of the state |

## Worked example: pricing an American option is optimal stopping

At each exercise date, the holder of an American option compares exercising now against continuing to hold:

```
V_t = max(payoff_t, E[discount · V_{t+1} | S_t])
```

This is a finite-horizon stochastic DP — backward induction from expiry, exactly like the finite-horizon case above, except the action space is binary (exercise / continue) and the state is the underlying asset's price.

**On a binomial lattice**, the conditional expectation `E[V_{t+1} | S_t]` is exact at every node (only two possible next prices, with known risk-neutral probabilities), so backward induction alone gives the price with no sampling noise:

```python
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
```

**With Monte Carlo simulated paths**, that conditional expectation is *unknown* — you only have sampled future outcomes along each path, not the true distribution. Longstaff & Schwartz (2001, *Review of Financial Studies* 14:113–147) estimate `E[continuation | S_t]` with a cross-sectional least-squares regression of realized discounted cash flows onto a low-order polynomial in the current price, using only in-the-money paths:

```python
def american_put_lsm(S0, K, r, sigma, T, steps=50, n_paths=100_000, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / steps
    Z = rng.standard_normal((n_paths // 2, steps)); Z = np.vstack([Z, -Z])  # antithetic
    S = np.exp(np.log(S0) + np.cumsum((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z, axis=1))
    cash = np.maximum(K - S[:, -1], 0)
    for t in range(steps - 2, -1, -1):           # backward induction, same direction as the lattice
        cash *= np.exp(-r * dt)
        itm = K - S[:, t] > 0                    # regress on in-the-money paths only
        if not itm.any(): continue
        x = S[itm, t] / K
        X = np.column_stack([np.ones_like(x), x, x**2])       # basis functions
        beta, *_ = np.linalg.lstsq(X, cash[itm], rcond=None)  # fits E[continuation | S_t]
        exercise = K - S[itm, t]
        ex_now = exercise > X @ beta
        cash[np.where(itm)[0][ex_now]] = exercise[ex_now]
    return np.exp(-r * dt) * cash.mean()
```

This is approximate DP in miniature: the exact Bellman backup (an expectation under a known model) is replaced by a regression fit to Monte Carlo samples, exactly the substitution the ADP table above describes generically.

**Validated result** (LS 2001's own Table 1 test case: S0=36, K=40, r=6%, σ=20%, T=1): the binomial lattice converges to **4.4867** (identical at 2,000 and 4,000 steps). LSM with 50 exercise dates and 100,000 paths gives **4.4678** — biased low, for two independent, expected reasons:

1. It prices a **Bermudan** option with 50 discrete exercise dates, not continuous exercise (continuous exercise is worth at least as much, since it's a strict superset of exercise opportunities).
2. The regressed exercise policy is only an *approximation* to the truly optimal one, and following a suboptimal stopping policy can only match or underperform the optimal policy's value — it can't beat it.

For a clean lower-bound estimate, fit the regression on one independent set of paths and price on another (removes look-ahead bias from fitting and pricing on the same data). For a rigorous upper bound to sandwich the true price, see the Andersen-Broadie duality method (not implemented here).

## When to reach for this family

The problem involves a sequence of decisions made under uncertainty, where each decision's value depends on an expectation over random future outcomes — MDPs, optimal stopping, inventory/pricing policies, American-style derivatives. If the state space is small enough to tabulate exactly, use value/policy iteration or backward induction directly. If it isn't — continuous state, or too many dimensions — that is exactly the signal to bring in Monte Carlo sampling (LSM-style regression, or the broader Monte Carlo skill in this repository for the sampling side of the problem).
