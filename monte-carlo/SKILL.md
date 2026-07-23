---
name: monte-carlo
description: |
  Full Monte Carlo simulation skill — use whenever the user wants to estimate probabilities, model uncertainty, price derivatives, run stochastic simulations, or apply sampling-based methods to any problem. Triggers on: "Monte Carlo", "simulate", "probability of X happening", "run N trials", "option pricing", "VaR", "uncertainty analysis", "MCMC", "Bayesian sampling", "random sampling", "stochastic model", "confidence interval via simulation", or any question that reduces to "what's the distribution of this output given uncertain inputs". Even if the user doesn't say "Monte Carlo" explicitly — if the problem involves randomness, distributions, or repeated sampling to get an answer, this skill applies.
---

# Monte Carlo Simulation Skill

Monte Carlo methods use repeated random sampling to get numerical answers to problems that are hard (or impossible) to solve analytically. The core insight: if you can't integrate a function or calculate a probability directly, you can *sample* it enough times and the statistics converge to the truth.

**Environment for running the reference scripts below:** they need `numpy`+`scipy` (some also `matplotlib`); the system `python3` doesn't have them. Use the venv at `../monte-carlo-workspace/reference-scripts/.venv/bin/python3` (relative to this file) — see that directory's own `README.md` for setup. If installed under an agent's skills folder (e.g. `~/.claude/skills/`), that resolves to `~/.claude/skills/monte-carlo-workspace/reference-scripts/.venv/`; keep one shared venv there rather than a separate one per tool.

## Step 1: Identify the Problem Type

Before writing any code, classify the problem — it determines the approach:

| Type | Signal phrases | Approach |
|------|---------------|----------|
| **Estimation** | "estimate π", "probability of X", "what fraction…" | Simple sampling, law of large numbers |
| **Numerical integration** | "integrate f(x)", "expected value of…", "area under curve" | Importance sampling or crude MC |
| **Financial** | "option price", "VaR", "portfolio risk", "cash flow" | GBM / correlated paths |
| **Multi-asset portfolio** | "portfolio of X and Y", "correlated assets", VaR across more than one holding | Correlated GBM via Cholesky — see `references/correlated-assets.md` |
| **Dependence with non-Gaussian marginals or tail asymmetry** | "correlated crashes", "joint extreme events", non-normal marginals need correlating | Copulas (Gaussian/t/Clayton) — see `references/copulas.md` |
| **Fat-tailed / extreme-move financial** | "tail risk", "crash scenario", "how bad could it get", GBM VaR looks too low vs. history | Alpha-stable distribution fit — see `references/stable-distributions.md` |
| **Skewed/fat-shoulder returns, finite variance** | Non-Gaussian returns without wanting infinite-variance tails | Variance-Gamma — see `references/variance-gamma.md` |
| **Deep-tail VaR/CVaR (99.9%+)** | "1-in-1000-day", "worst case in N years", GBM/stable/VG tail feels like extrapolating on faith | Extreme Value Theory (GEV/GPD) — see `references/extreme-value-theory.md` |
| **Time series forecasting** | "forecast X over time", "sensor/demand/ops trend", "prediction interval for a series" | ARIMA/SARIMA + residual bootstrap MC — see `references/time-series.md` |
| **Scientific/engineering** | "uncertainty propagation", "reliability", "sensitivity" | Input distribution sampling → propagate through model |
| **MCMC / Bayesian** | "posterior distribution", "fit a model", "parameter estimation" | Metropolis-Hastings or recommend PyMC/Stan; for correlated/high-dim posteriors see `references/hmc-nuts.md` |
| **Hidden/latent state tracking over time** | "filter volatility", "track a hidden state", nonlinear/non-Gaussian state-space model | Particle filter (bootstrap/SIR) — see `references/particle-filters.md` |

Ask the user one clarifying question if the type is genuinely ambiguous. If it's clear, proceed.

## Step 2: Design the Simulation

Walk through this mentally before coding:

1. **What are the random inputs?** (distributions, correlations)
2. **What is the quantity of interest?** (the scalar or vector to estimate)
3. **What's the estimator?** (the function that maps one sample → one value)
4. **How many samples?** Start with 10,000–100,000. For financial/scientific work, convergence analysis matters (run at N=1k, 10k, 100k and check stability).
5. **Variance reduction needed?** For rare events (<1% probability) or high-variance problems, plain sampling is slow — mention antithetic variates, control variates, or importance sampling.

## Step 3: Implement

### Standard Python template (works for most problems)

```python
import numpy as np

rng = np.random.default_rng(seed=42)  # reproducible; drop seed for production
N = 100_000

# --- define your random inputs here ---
# x = rng.uniform(0, 1, N)
# y = rng.normal(mu, sigma, N)

# --- compute the quantity of interest for each sample ---
# results = f(x, y)

# --- aggregate ---
estimate = results.mean()
std_err = results.std(ddof=1) / np.sqrt(N)  # ddof=1: unbiased sample variance
ci_95 = (estimate - 1.96 * std_err, estimate + 1.96 * std_err)

print(f"Estimate: {estimate:.4f}")
print(f"95% CI:   [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")
print(f"Std err:  {std_err:.6f}")
```

The symmetric normal CI above (`estimate ± 1.96*std_err`) relies on the CLT and is fine for a plain mean over enough samples, but it is the wrong tool for several common cases — don't apply it blindly:
- **Proportions/probabilities** (Bernoulli outcomes), especially near 0 or 1: use a Wilson or Beta interval instead of the normal approximation.
- **Quantiles** (VaR, percentiles): use a bootstrap CI — the normal-approximation formula above doesn't apply to order statistics.
- **Skewed payoffs** (option prices, insurance losses): bootstrap or batch-means; a symmetric interval on a heavily skewed distribution misstates the tails.
- **MCMC output**: use posterior credible intervals (quantiles of the trace), not an iid-sample CI — MCMC draws are autocorrelated, so the `/√N` formula overstates precision unless you use the effective sample size.

Always:
- Report a **confidence interval**, not just a point estimate — and pick the CI method that matches what you're estimating (see above)
- Set a **random seed** when reproducibility matters — for parallel runs, derive per-worker seeds from `np.random.SeedSequence(seed).spawn(n_workers)` rather than reusing one seed or incrementing by hand, so streams don't silently overlap
- Use **vectorised numpy** (not Python loops) — 100× faster

See `references/financial.md` for GBM / option pricing patterns.  
See `references/mcmc.md` for Bayesian / MCMC patterns.  
See `references/variance-reduction.md` for antithetic variates, control variates, importance sampling, stratified sampling, and quasi-Monte Carlo (Sobol/Halton).  
See `references/time-series.md` for ARIMA/SARIMA + residual-bootstrap Monte Carlo forecasting patterns, with a reviewed, importable reference implementation.  
See `references/stable-distributions.md` for alpha-stable (fat-tailed) return modeling when GBM's Gaussian tails understate real risk, with a reviewed, importable reference implementation.  
See `references/correlated-assets.md` for correlated multi-asset GBM (portfolio VaR/CVaR across assets that move together), with a reviewed, importable reference implementation.  
See `references/parameterization-conventions.md` for numpy vs scipy distribution parameter conventions (shape/scale, loc/scale gotchas) — check before writing new sampling code for a distribution not already used elsewhere in this skill.  
See `references/copulas.md` for copula-based dependence modeling (Gaussian/t/Clayton) when marginals aren't Gaussian/lognormal or tail dependence needs to be asymmetric — with a reviewed, importable reference implementation.  
See `references/hmc-nuts.md` for Hamiltonian Monte Carlo and the No-U-Turn Sampler — gradient-based MCMC for correlated/high-dimensional posteriors where Metropolis-Hastings' random walk struggles, with a reviewed, importable reference implementation and measured effective-sample-size comparisons.  
See `references/variance-gamma.md` for the Variance-Gamma process — a finite-variance, skewed/fat-shouldered alternative to GBM sitting between Gaussian and alpha-stable, with a reviewed, importable reference implementation.  
See `references/extreme-value-theory.md` for GEV/GPD extreme value theory — deep-tail VaR/CVaR (99.9%+) via peaks-over-threshold, valid regardless of the parent return distribution, with a reviewed, importable reference implementation validated against a known ground-truth distribution.  
See `references/particle-filters.md` for bootstrap particle filters — recovering a hidden state (e.g. stochastic volatility) from noisy observations when the state-space model is nonlinear/non-Gaussian, with a reviewed, importable reference implementation validated against an exact Kalman filter.

## Step 4: Validate the Result

A Monte Carlo result without validation is just a number. Do at least one of:

- **Analytical check**: if a closed-form answer exists, verify the MC matches it (e.g., Black-Scholes for European options, Beta/Binomial for simple probabilities)
- **Convergence plot**: plot estimate vs. N — should stabilise as N grows
- **Confidence interval sanity**: the 95% CI should contain the truth ~95% of the time if you ran many trials
- **Relative standard error**: aim for `std_err / abs(estimate) < 0.01` — only meaningful when the estimate is safely away from zero. For a quantity that's expected to be near zero (a hedged P&L, a difference of two similar values), use an absolute error target instead; the ratio blows up or is meaningless there.

## Step 5: Present Results Clearly

Good MC output includes:
1. The estimate ± margin of error (or CI)
2. Number of samples used
3. Any key assumptions (distributions chosen, correlations assumed)
4. A brief note on what would make the estimate more accurate (more samples, better input distributions, variance reduction)

If producing a chart: histogram of outcomes + vertical line at mean + shading for CI is the standard.

---

## Domain Quick-Reference

### General estimation / integration
- Use vectorised sampling with `numpy`
- For multidimensional integrals, plain MC beats quadrature above ~4 dimensions
- Convergence rate is O(1/√N) regardless of dimension — this is MC's superpower
- For smooth integrands in low-to-moderate dimensions, quasi-Monte Carlo (Sobol sequences via `scipy.stats.qmc`) converges faster than plain MC — see `references/variance-reduction.md`

### Financial (see `references/financial.md` for full patterns)
- Geometric Brownian Motion for stock paths: `S_t = S_0 * exp((μ - σ²/2)t + σ√t * Z)`
- For options: simulate terminal price, compute payoff, discount
- For path-dependent (Asian, barrier): simulate full path, not just terminal

### Time series forecasting (see `references/time-series.md` for full patterns)
- Fit ARIMA/SARIMA for trend + (single) seasonal structure, then Monte Carlo forward by resampling the fitted model's own in-sample residuals (bootstrap) instead of assuming Gaussian innovations
- Reviewed, importable implementation: `../monte-carlo-workspace/reference-scripts/arima_residual_bootstrap.py`
- Multiple seasonalities → MSTL/TBATS instead of SARIMA; intermittent/zero-heavy data → Croston/TSB/IMAPA; volatility clustering → ARIMA-GARCH — different deterministic core, same residual-bootstrap wrapper idea
- Always validate via rolling-window walk-forward coverage backtest (does the nominal interval actually hold at that rate out-of-sample), with a standard error on the coverage estimate — small window counts (~10-15) have real sampling noise, don't over-read a small coverage gap as a real effect

### Scientific / engineering
- Identify uncertain inputs → assign distributions (ask user or use normal/uniform as defaults)
- Propagate through deterministic model `y = f(x_1, ..., x_n)`
- Sobol indices for sensitivity analysis if user asks "which input matters most"

### MCMC / Bayesian
- For simple posteriors: implement Metropolis-Hastings inline
- For real Bayesian inference problems: recommend PyMC (`pip install pymc`) or Stan — don't reimplement MCMC from scratch unless the user explicitly wants to learn
- Key diagnostics: trace plot, R-hat < 1.01, effective sample size > 400

---

## Common Pitfalls to Avoid

- **Using `random.random()` in a loop**: 100× slower than numpy vectorisation
- **No confidence interval**: a point estimate alone is misleading
- **Too few samples**: MC error scales as `sample_std / sqrt(N)`, not a fixed percentage — the "3% at 1k, 0.1% at 1M" rule of thumb only holds for a bounded 0/1 outcome near p=0.5 (where std ≈ 0.5). For other quantities, run a small pilot (N=1k) to get `s = results.std(ddof=1)`, then solve for the N a real decision needs: `N_required = (1.96 * s / target_half_width) ** 2`
- **Correlated inputs treated as independent**: always ask if inputs are correlated before assuming independence
- **Forgetting to discount cash flows** in financial simulations
- **Not checking convergence**: always plot estimate vs N for anything that will be used in a real decision
