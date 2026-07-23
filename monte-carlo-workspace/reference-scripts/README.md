# Monte Carlo Reference Scripts

Importable, adversarially-reviewed Python implementations backing the [`monte-carlo` skill](../../monte-carlo/README.md). Each module is documented in more depth in that skill's `references/*.md` files — this README is just the setup + index.

## Setup

Needs `numpy` + `scipy` (some modules also `matplotlib`, `pandas`, `statsmodels`); the system `python3` typically won't have these. Create a venv once and reuse it:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy matplotlib pandas statsmodels
```

Run any script directly, or import it in your own code:

```bash
.venv/bin/python3 gbm_stock_sim.py --help
```

```python
import sys
sys.path.insert(0, "/path/to/reference-scripts")
from copula_sampling import t_copula_sample, apply_marginals
```

## Modules

- **`gbm_stock_sim.py`** — Single-asset Geometric Brownian Motion price simulation and option pricing (terminal-price shortcut, not full path, when only the terminal value is needed).
- **`correlated_gbm.py`** — Multi-asset GBM via Cholesky-correlated shocks for portfolio VaR/CVaR across assets that move together; PSD validation via eigenvalues (not Cholesky, which wrongly rejects valid singular matrices).
- **`stable_distribution.py`** — Alpha-stable (Lévy stable) fitting and simulation for fat-tailed returns beyond what GBM/Gaussian models capture; guards against a known scipy MLE boundary-collapse failure mode.
- **`variance_gamma.py`** — Variance-Gamma process, a finite-variance skewed/fat-shouldered alternative to GBM, built from a from-scratch Gamma-subordinated Brownian Motion construction (`scipy.stats.genhyperbolic` is numerically fragile at this boundary case).
- **`copula_sampling.py`** — Gaussian/Student-t/Clayton copula sampling: models dependence structure independent of each asset's own marginal distribution (`apply_marginals()` composes with any ppf, including the other modules here).
- **`extreme_value.py`** — GEV/GPD extreme value theory via peaks-over-threshold for deep-tail (99.9%+) VaR/CVaR, valid regardless of the parent return distribution; includes bootstrap CIs since the tail-shape parameter is high-variance to estimate.
- **`hmc_nuts.py`** — Hamiltonian Monte Carlo and the No-U-Turn Sampler for correlated/high-dimensional Bayesian posteriors where Metropolis-Hastings random-walk proposals struggle.
- **`particle_filter.py`** — Bootstrap particle filter for recovering a hidden state (e.g. stochastic volatility) from noisy observations in nonlinear/non-Gaussian state-space models; validated against an exact Kalman filter on a linear-Gaussian special case.
- **`arima_residual_bootstrap.py`** — ARIMA/SARIMA time-series forecasting with Monte Carlo forward simulation via residual bootstrap instead of assuming Gaussian innovations.

## Every module has at least one real, adversarially-found bug fixed

Each script was built, then reviewed independently for the statistical-correctness failure modes Monte Carlo code is prone to — silent bias, boundary-value collapse, scipy sign/parameterization mismatches, tail-clipping from performance shortcuts, double-counted noise sources. All were run and checked against a known-good reference (an exact formula, an independent oracle model, or an empirical convergence check) before being trusted, not accepted on a single "looks right" pass.

## License

MIT — see `../../monte-carlo/LICENSE`.
