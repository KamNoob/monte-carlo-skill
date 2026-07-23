# Financial Monte Carlo Reference

## Geometric Brownian Motion (GBM)

Standard model for equity price paths:

```python
import numpy as np

def simulate_gbm_paths(S0, mu, sigma, T, dt, N_paths, seed=None):
    """
    S0: initial price
    mu: annualised drift (risk-neutral: use risk-free rate r)
    sigma: annualised volatility
    T: time horizon in years
    dt: requested time step (e.g., 1/252 for daily) — rescaled below so
        n_steps * dt_actual == T exactly. `int(T / dt)` silently truncates
        the horizon whenever dt doesn't divide T evenly (e.g. T=1, dt=0.3
        would simulate 0.9 years, not 1); recomputing dt from a rounded
        step count avoids that.
    N_paths: number of simulation paths
    """
    rng = np.random.default_rng(seed)
    n_steps = max(1, round(T / dt))
    dt_actual = T / n_steps

    Z = rng.standard_normal((N_paths, n_steps))
    log_returns = (mu - 0.5 * sigma**2) * dt_actual + sigma * np.sqrt(dt_actual) * Z
    
    log_paths = np.cumsum(log_returns, axis=1)
    paths = S0 * np.exp(log_paths)
    
    # prepend S0
    paths = np.hstack([np.full((N_paths, 1), S0), paths])
    return paths  # shape: (N_paths, n_steps+1)
```

## European Option Pricing

```python
def price_european_option(S0, K, r, sigma, T, option_type='call', N=100_000, seed=42):
    """Standard Black-Scholes Monte Carlo."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(N)
    
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)
    
    price = np.exp(-r * T) * payoffs.mean()
    std_err = np.exp(-r * T) * payoffs.std() / np.sqrt(N)
    
    return price, std_err

# Validate against Black-Scholes analytical formula:
from scipy.stats import norm
def bs_call(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
```

## Asian Option (path-dependent)

```python
def price_asian_call(S0, K, r, sigma, T, dt=1/252, N=50_000, seed=42, include_initial=True):
    """
    include_initial: whether the average includes S0 (t=0) or only the
    n_steps post-start grid points. Averaging convention changes the price
    materially for short-dated options — state which one you used.
    """
    paths = simulate_gbm_paths(S0, r, sigma, T, dt, N, seed)
    avg_prices = paths.mean(axis=1) if include_initial else paths[:, 1:].mean(axis=1)
    payoffs = np.maximum(avg_prices - K, 0)
    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    std_err = discounted.std(ddof=1) / np.sqrt(N)
    return price, std_err, (price - 1.96 * std_err, price + 1.96 * std_err)
```

## Barrier Option

```python
def price_down_and_out_call(S0, K, B, r, sigma, T, dt=1/252, N=50_000, seed=42):
    """
    Knocked out if price ever falls below barrier B.

    WARNING — discrete monitoring bias: this only checks the simulated grid
    points, so a path that dips below B *between* dates and recovers is
    scored as "survived" when true continuous monitoring would knock it
    out. This systematically overprices the option. Mitigate with a finer
    dt, or apply the Brownian-bridge continuity correction (Broadie,
    Glasserman & Kou 1997): shift the barrier to
    B * exp(-0.5826 * sigma * sqrt(dt)) (down-and-out) before comparing.
    """
    paths = simulate_gbm_paths(S0, r, sigma, T, dt, N, seed)
    survived = (paths.min(axis=1) > B)
    S_T = paths[:, -1]
    payoffs = np.maximum(S_T - K, 0) * survived
    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    std_err = discounted.std(ddof=1) / np.sqrt(N)
    return price, std_err, (price - 1.96 * std_err, price + 1.96 * std_err)
```

## Value at Risk (VaR) and CVaR

```python
def portfolio_var_cvar(weights, mu_vec, cov_matrix, portfolio_value, horizon_days, 
                        confidence=0.99, N=100_000, seed=42):
    """
    weights: array of portfolio weights
    mu_vec: annualised expected returns
    cov_matrix: annualised covariance matrix
    """
    rng = np.random.default_rng(seed)
    T = horizon_days / 252
    
    # Multivariate normal returns
    returns = rng.multivariate_normal(mu_vec * T, cov_matrix * T, N)
    portfolio_returns = returns @ weights
    portfolio_pnl = portfolio_value * portfolio_returns
    
    var = -np.percentile(portfolio_pnl, (1 - confidence) * 100)
    tail = portfolio_pnl[portfolio_pnl <= -var]
    cvar = -tail.mean()

    expected_tail_count = N * (1 - confidence)
    if expected_tail_count < 30:
        print(f"WARNING: only ~{expected_tail_count:.0f} samples in the tail "
              f"at {confidence:.0%} confidence — CVaR estimate is noisy. "
              f"Increase N or lower confidence, and consider bootstrapping "
              f"a CI around both var and cvar.")

    return var, cvar
```

Caveat: this assumes multivariate-normal returns. Real portfolio returns have fatter tails than a Gaussian, which understates true tail risk — for anything decision-critical, prefer a Student-t, alpha-stable (see `stable-distributions.md`), or historical-bootstrap return model, or say explicitly that Gaussian VaR is a lower bound on real risk. Note this function already handles cross-asset correlation via `cov_matrix` — for a GBM-path (not single-period) treatment of correlated assets, e.g. multi-day horizons or per-asset percentile reporting rather than portfolio-only, see `correlated-assets.md`.

## Cash Flow Simulation

```python
def simulate_project_npv(base_cashflows, uncertainty_pct, discount_rate, N=10_000, seed=42):
    """
    base_cashflows: array of EXPECTED cash flows per period [CF0, CF1, ..., CFn]
    uncertainty_pct: fractional uncertainty (e.g., 0.2 = ±20%), applied as
        the lognormal shape parameter sigma
    discount_rate: per-period discount rate
    """
    rng = np.random.default_rng(seed)
    n_periods = len(base_cashflows)

    # A lognormal(mu, sigma) multiplier has mean exp(mu + sigma^2/2), not
    # exp(mu) — using mu=0 here would silently inflate every cash flow's
    # expectation by exp(uncertainty_pct^2 / 2) (e.g. +2% at 20% uncertainty,
    # +13% at 50%), so base_cashflows would stop being the *expected* value.
    # Set mu = -sigma^2/2 so E[noise] == 1 and the mean is preserved.
    sigma = uncertainty_pct
    noise = rng.lognormal(-0.5 * sigma**2, sigma, (N, n_periods))
    cf_simulations = base_cashflows * noise
    
    periods = np.arange(n_periods)
    discount_factors = (1 + discount_rate) ** (-periods)
    
    npvs = (cf_simulations * discount_factors).sum(axis=1)
    
    return {
        'mean_npv': npvs.mean(),
        'std_npv': npvs.std(),
        'prob_positive': (npvs > 0).mean(),
        'p5': np.percentile(npvs, 5),
        'p95': np.percentile(npvs, 95),
        'distribution': npvs
    }
```

## Key Assumptions to State

Always be explicit about:
- Risk-neutral vs. real-world measure (use r for pricing, μ for risk)
- Dividend yield if applicable (`mu → r - q`)
- Whether volatility is constant (plain GBM) or needs stochastic vol (Heston)
- Correlation structure for multi-asset problems
