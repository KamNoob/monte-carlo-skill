# Alpha-Stable (Levy Stable) Distribution Reference

## When to reach for this instead of GBM/lognormal

`references/financial.md`'s GBM model assumes Gaussian log-returns. Real daily
equity/FX returns are fatter-tailed than that — large moves happen more often
than a Gaussian predicts. The alpha-stable family is the standard heavy-tail
alternative: a 4-parameter generalisation of the normal distribution that
includes it as a special case (alpha=2) and lets tail heaviness be estimated
from data instead of assumed away.

Use it when:
- The user asks for tail risk / VaR / CVaR on an asset where fat tails
  matter more than the median outcome (options, leveraged positions, crash
  scenarios)
- A GBM backtest's realized extreme moves fall outside its simulated
  distribution more often than the nominal confidence level predicts
  (a coverage-check failure, same idea as `references/time-series.md`'s
  walk-forward backtest)

Don't use it as the default financial model — it's slower to fit, harder to
reason about (see infinite-variance caveat below), and for most day-to-day
"where might this price land in a week" questions plain GBM is the right
tool. Reach for stable when tail behavior is the actual question.

## Parameters

| Param | Range | Meaning |
|-------|-------|---------|
| alpha | (0, 2] | Tail index / stability. alpha=2 is exactly Gaussian. Lower alpha = heavier power-law tails (P(\|X\|>x) ~ x^-alpha). Empirical equity/FX daily returns typically fit alpha ≈ 1.5-1.9. |
| beta  | [-1, 1] | Skewness. 0 = symmetric. |
| loc, scale | — | Location and scale, analogous to mean/std but **not equal to them** — see below. |

**Critical gotcha — infinite variance:** for alpha < 2 the population variance
is infinite (undefined entirely for alpha <= 1). This isn't a numerical
quirk to work around — it's the actual mathematical content of a
heavy-tailed model, and it means:
- Never report sample mean/std of a stable-distributed series as if it
  estimates a real population mean/std the way it does for GBM. It doesn't
  converge to anything as sample size grows.
- Use quantile-based risk measures (VaR, CVaR from simulated/empirical
  percentiles) instead of moment-based ones (Sharpe ratio, `mu ± 1.96*sigma`
  CIs). This is the same guidance as `SKILL.md`'s CI-method table, just with
  a mandatory case here instead of a "which CI fits this stat" judgment call.

**Parameterization gotcha:** scipy supports two internal parameterizations
(`S0`, `S1`, default `S1`), which disagree at alpha=1 in particular and
give different `loc` values under skew (confirmed empirically: the same
skewed data fit under S0 vs S1 gave loc differing by ~0.0036, alpha/beta/
scale unaffected). `scipy.stats.levy_stable` is a **process-wide
singleton** — `dist.parameterization = "S0"` mutates global state for
every caller in the process, not a local view. The reference script wraps
every fit/rvs call in a context manager that saves the previous
`parameterization`/`pdf_default_method` and restores them afterward, so it
can't leave global scipy state different from how it found it. Don't
replace that with a bare `dist.parameterization = "S0"` assignment even
for a quick script — S0 is continuous in all four parameters (S1 has a
discontinuity at alpha=1 that has caused real scipy bugs, see scipy#20821),
so S0 is still the right choice, just scope the mutation.

## Fitting

```python
from scipy import stats
stats.levy_stable.parameterization = "S0"
alpha, beta, loc, scale = stats.levy_stable.fit(returns)
```

scipy's `fit()` uses McCulloch's (1986) quantile estimator (reads the
5th/25th/50th/75th/95th percentiles, looks up alpha/beta from precomputed
tables) as a starting point, then refines by MLE. The reference script
enforces a 250-observation floor — empirically even 500 points recovered
beta with ~40% error against a known true value, so treat any fit as an
estimate with real uncertainty, not a precise readout, regardless of
sample size.

**MLE can silently converge to a degenerate boundary fit.** Observed
directly: fitting 100 synthetic points drawn from alpha=1.7 returned
alpha=2.0, beta=-1.0 — a syntactically valid result that is actually a
failed optimization pinned at the edge of the parameter space, not a real
answer. This is a documented scipy MLE failure mode, not a data problem.
The reference script rejects fits within 1e-3 of alpha in {0, 2} or
abs(beta)=1 rather than returning them silently — don't remove that check
when reusing this code, and don't add moment-based ("does this look
Gaussian-ish") heuristics as a substitute, since a boundary fit look
exactly as clean as a real one until you check the parameters themselves.

**Known slow/fragile spots** (from scipy's own docs, confirmed empirically):
- The default 'piecewise' pdf evaluation (Nolan's method) does numerical
  integration per point, so MLE refinement over many pdf calls is much
  slower than fitting a normal or lognormal — measured **~70 seconds for
  500 points** on reference hardware (scipy 1.18), scaling roughly linearly
  with sample size. This is not "seconds," it's real wall-clock cost: for a
  year of daily returns (~250 points) budget well under a minute, but for
  anything in the thousands, expect minutes. Never call `fit()` in a tight
  loop (e.g. once per rolling window in a walk-forward backtest) without
  timing it first — it will dominate runtime.
- MLE isn't guaranteed to converge when alpha <= 1 combined with the FFT pdf
  method — stick with the default piecewise method rather than opting into
  `pdf_default_method='fft-simpson'` for fitting.

## Sampling (Chambers-Mallows-Stuck)

`scipy.stats.levy_stable.rvs(alpha, beta, loc=, scale=, size=, random_state=)`
implements the Chambers-Mallows-Stuck (1976) algorithm internally — a
closed-form transform of a uniform and an independent exponential draw, no
rejection sampling or numerical inversion needed, so `rvs()` itself is fast
even though `fit()`/`pdf()` are not. Pass a `np.random.Generator` via
`random_state` for the same seeding conventions used elsewhere in this
skill.

Stable variables are additive: a sum of `n` iid `stable(alpha, beta, scale)`
draws is itself `stable(alpha, beta, n**(1/alpha) * scale)` (not
`sqrt(n)*scale` as for the Gaussian case, except when alpha=2 where they
coincide) — this is what makes it valid to sum sampled log-returns into a
cumulative log-return over a horizon, same as the CLT-based sum for GBM, but
the scale grows faster than `sqrt(n)` for alpha<2.

## Reviewed reference implementation

`../../monte-carlo-workspace/reference-scripts/stable_distribution.py`
— fits alpha-stable to a returns series, Monte Carlo simulates a horizon
cumulative return under both the stable fit and a Gaussian fit on the same
data, and reports VaR/CVaR from each so the tail-risk gap a Gaussian model
misses is directly visible. Run `--demo` for a synthetic alpha=1.7 example,
or `--returns-csv path.csv` for real data.

## Validation

- **Alpha sanity check**: if the fitted alpha lands outside roughly [1, 2]
  for daily financial returns, treat it as a sign of a data problem (outliers,
  regime change, too few points) before trusting the tail-risk numbers —
  alpha well below 1 implies undefined mean, which is implausible for most
  real return series over normal market conditions.
- **Compare to the Gaussian VaR/CVaR on the same data** (what the reference
  script does) — the ratio is the concrete answer to "how much is a GBM
  model underestimating tail risk here."
- **Coverage backtest** if used for forecasting, same walk-forward approach
  as `references/time-series.md`.
