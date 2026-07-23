# Time Series + Monte Carlo (ARIMA Residual Bootstrap)

## When this applies

Use when a problem needs uncertainty simulated **forward through a fitted time-series model**, rather than around a fixed drift (GBM) or a static distribution — e.g. forecasting a real, autocorrelated/seasonal series (sensor readings, ops metrics, demand, weather) and wanting Monte Carlo paths/intervals that respect the series' own trend + seasonality, not just i.i.d. noise around a point estimate.

Not for: pure option/portfolio pricing (use `references/financial.md` — GBM), Bayesian parameter inference (use `references/mcmc.md`), or a one-off static uncertainty propagation with no time dimension.

## The method: residual bootstrap

Standard, established technique (same as R's `forecast::simulate.Arima(bootstrap=TRUE)`): fit ARIMA/SARIMA to get the deterministic trend+seasonal structure, then instead of assuming Gaussian innovations for the forward simulation, **resample the model's own in-sample residuals with replacement** to drive each Monte Carlo step. Gives an empirical, non-parametric forecast distribution — matters when real residuals are fat-tailed/skewed, which is common (Gaussian-assumption intervals understate tail risk the same way plain-normal VaR does in `financial.md`).

A hardened, reviewed reference implementation lives at:
`../../monte-carlo-workspace/reference-scripts/arima_residual_bootstrap.py`

It implements, as importable functions:
- `fit_best_sarima(series, period, candidate_orders=None)` — small AIC grid search over SARIMA orders (not full auto-ARIMA, but not hand-waved either)
- `bootstrap_forecast_paths(fit, horizon, n_paths, rng)` — hand-rolled state-space recursion (statsmodels' `simulate(state_shocks=...)` doesn't support per-path custom shocks), vectorized across paths per step, drawing shocks via `rng.choice(resid_pool, size=n_paths, replace=True)` at each horizon step
- `naive_gaussian_forecast(fit, horizon, alpha)` — statsmodels' standard Gaussian-assumption interval, for comparison
- `percentile_interval(paths, conf_level)` — correct percentile-based interval on simulated paths (never mean±z·std on a bootstrap distribution — that throws away the whole point of not assuming Gaussian)
- `rolling_backtest(series, period, horizon, n_paths, min_train, step, conf_level, seed)` — walk-forward coverage validation with no lookahead, reports empirical coverage **and its standard error** for both bootstrap and naive methods

Import these directly rather than re-deriving the state-space recursion — it was reviewed by `mc-model-reviewer` and had two real ordering bugs caught and fixed (shock-timing, predicted/realized state indexing) before it was verified to exactly reproduce statsmodels' own deterministic forecast at zero-shock.

## Non-obvious pitfalls specific to this technique

- **Burn-in contamination**: diffuse/approximate-diffuse state initialization leaves the first several residuals badly distorted (can be 15-20 std devs off — a real bug found in review). Don't resample from `results.resid` raw; use `max(order-based heuristic, results.loglikelihood_burn)` to decide how many initial residuals to drop, matching what the reference script does.
- **Time-invariance assumption**: the hand-rolled recursion flattens the state-space matrices (Z/T/R) to their first time-slice. Fine for plain SARIMA; silently wrong if exogenous regressors or time-varying terms are added later without adding a `ssm.time_invariant` guard.
- **Coverage validation needs a standard error, not just a point estimate**: a rolling backtest with ~10-15 windows has real sampling noise on the coverage estimate (the within-window horizon points are correlated, not independent trials) — a small coverage gap between bootstrap and naive Gaussian is often statistically indistinguishable from noise at typical window counts. Report the SE, don't over-read a few-point gap as a real effect.
- **Domain-specific core swap**: plain ARIMA/SARIMA handles a trend + a single seasonal period. Swap the deterministic core (same residual-bootstrap wrapper idea, different mechanics) for:
  - **Multiple seasonalities** (e.g. daily+weekly cycles in hourly data): MSTL or TBATS — but these aren't drop-in replacements, they have their own internal error/state-space structure, not "ARIMA with extra steps."
  - **Intermittent/zero-heavy demand**: Croston/TSB/IMAPA — Gaussian-residual ARIMA can predict negative counts and misses intermittent-arrival structure; count-appropriate alternatives are the right base.
  - **Volatility clustering** (financial-style data, some real-world series): ARIMA-GARCH — GARCH models the conditional variance, Monte Carlo simulates through both mean and variance jointly. Only worth the added complexity when clustering is actually present in the residuals, not by default.

## Validated result (as of 2026-07-21)

Tested on synthetic data (near-Gaussian, skew 0.94/kurtosis 2.38) and real hourly temperature data (skew -0.34/kurtosis 6.14, genuinely fatter-tailed): point forecasts showed a real, meaningful accuracy gain over naive seasonal-persistence (21% MAE improvement on the real series). Interval-calibration advantage of bootstrap over naive Gaussian was inconclusive on both datasets — coverage gaps were within 1-1.5x the standard error at the window counts tested (~10-12). Treat the bootstrap's calibration edge as theoretically expected but not yet empirically confirmed at useful confidence; the point-forecast improvement from ARIMA itself is the more solid, demonstrated result.
