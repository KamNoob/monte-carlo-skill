# numpy / scipy Distribution Parameterization Conventions

This skill mixes `numpy.random.Generator` (fast vectorized sampling) and
`scipy.stats` (fitting, PDFs, less-common distributions). Their parameter
conventions mostly agree but diverge in specific, easy-to-miss ways. Two
real bugs already found via this exact gap (documented in
`stable-distributions.md` and `financial.md`'s NPV section) motivated
writing this down rather than trusting memory each time — verify against
official docs (links below) before shipping new sim code, don't assume.

## Quick-reference table

| Distribution | numpy call | numpy params | scipy call | scipy params | Agreement |
|---|---|---|---|---|---|
| Normal | `rng.normal(loc, scale)` | mean, std | `stats.norm(loc, scale)` | mean, std | Match |
| Lognormal | `rng.lognormal(mean, sigma)` | mean/sigma of the **underlying normal**, not the lognormal's own mean | `stats.lognorm(s, loc, scale)` | `s`=sigma of underlying normal, `scale`=exp(mean) | Different call shape, same underlying math — see gotcha below |
| Exponential | `rng.exponential(scale)` | scale (mean) = 1/rate | `stats.expon(loc, scale)` | scale (mean) = 1/rate | Match |
| Gamma | `rng.gamma(shape, scale)` | shape (k), scale (θ) | `stats.gamma(a, loc, scale)` | shape (a=k), scale (θ) | Match — both shape/scale, verified against scipy docs which explicitly note `scale = 1/beta` vs the shape/rate form |
| Weibull | `rng.weibull(a)` | shape only, **no scale param** | `stats.weibull_min(c, loc, scale)` | shape (c=a), scale | numpy has no built-in scale — must multiply manually: `scale * rng.weibull(a)` matches `stats.weibull_min.rvs(a, scale=scale)` (verified against scipy docs) |
| Beta | `rng.beta(a, b)` | shape a, shape b | `stats.beta(a, b, loc, scale)` | shape a, shape b | Match |
| Student's t | `rng.standard_t(df)` | df only, **no loc/scale** | `stats.t(df, loc, scale)` | df, loc, scale | numpy gotcha — see below |
| Poisson | `rng.poisson(lam)` | rate (lambda) | `stats.poisson(mu)` | rate (mu) | Match, different param name only |
| Binomial | `rng.binomial(n, p)` | trials, success prob | `stats.binom(n, p)` | trials, success prob | Match |
| Multivariate normal | `rng.multivariate_normal(mean, cov)` | mean vector, **covariance** matrix | `stats.multivariate_normal(mean, cov)` | mean vector, covariance matrix | Match — note this is covariance, not correlation; see `correlated-assets.md` for the correlation-matrix case |
| Alpha-stable | n/a (not in numpy) | — | `stats.levy_stable(alpha, beta, loc, scale)` | S0 vs S1 parameterization differ in `loc` under skew | See `stable-distributions.md` — do not use scipy's default (S1) without setting `.parameterization` explicitly |

## Gotchas worth internalizing

**Student's t has no `loc`/`scale` in numpy, but does in scipy.** Porting
code from `scipy.stats.t.rvs(df, loc=mu, scale=sigma)` to numpy naively as
`rng.standard_t(df)` silently drops the location/scale shift — the numpy
call only gives the standard (mean 0, scale 1) t-distribution. The correct
numpy equivalent is `mu + sigma * rng.standard_t(df, size=size)`. This is
exactly the kind of "looks like a straightforward swap" bug that survives
casual review because both functions run without error — the numpy version
just silently produces the wrong location and scale.

**Lognormal's `mean` parameter is not the lognormal's own mean.**
`rng.lognormal(mean, sigma)` — `mean` and `sigma` describe the
**underlying normal distribution**, not the lognormal's own mean/std. A
lognormal(mu, sigma) multiplier has actual mean `exp(mu + sigma**2/2)`, not
`exp(mu)`. This bug already hit `financial.md`'s NPV cash-flow simulator —
see that file's `simulate_project_npv` docstring for the concrete fix
(`mu = -sigma**2/2` to keep `E[noise] == 1`). Any new lognormal usage
should re-derive this rather than assume `mean` means what it sounds like.

**Weibull has no scale parameter in numpy.** `rng.weibull(a)` only
generates the 1-parameter (unit-scale) Weibull. For the standard
2-parameter form, scale manually: `scale * rng.weibull(a)`. Confirmed
against scipy docs that this exactly matches `stats.weibull_min.rvs(a,
scale=scale)`.

**Gamma's shape/scale vs shape/rate split is a common cross-library trap
in general** (R's `dgamma` defaults to rate, not scale) — numpy and scipy
both use shape/scale here (verified above), so no bug within this
skill's own code, but a value pulled from an R reference or a paper that
states "rate" needs inverting (`scale = 1/rate`) before passing to either
numpy or scipy.

**Covariance vs correlation matrix.** `rng.multivariate_normal` and
`stats.multivariate_normal` both take a **covariance** matrix (diagonal =
variances, not 1.0). `reference-scripts/correlated_gbm.py` (see
`correlated-assets.md`) instead takes a **correlation** matrix
(diagonal = 1.0) and applies per-asset volatility
separately via Cholesky — don't pass one where the other is expected; the
validation in `correlated_gbm.py`'s `validate_correlation_matrix` would
reject a real covariance matrix's non-unit diagonal, which is a useful
guard, but only for that specific script.

## How to verify a distribution not covered here

1. Check the numpy/scipy docs pages directly — don't rely on memory, the
   parameterization is exactly the kind of detail that's easy to misstate
   from familiarity with one library's convention.
2. Draw a moderate sample (10k+) with known parameters and check the
   sample mean/variance against the distribution's known closed-form
   moments (Wikipedia's infobox for the distribution usually has these) —
   a quick empirical sanity check catches a parameterization mismatch
   immediately, before it reaches production sim output.
3. If porting code between numpy and scipy calls for the same
   distribution, run both on the same seed conceptually (not
   bit-for-bit — they use different underlying algorithms) and compare
   moments, not just "does it run without error."
