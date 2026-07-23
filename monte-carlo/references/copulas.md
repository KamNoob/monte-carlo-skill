# Copula-Based Dependence Modeling Reference

## What a copula actually is, and when to reach for one

A copula is a joint distribution on `[0,1]^n` with uniform marginals —
it captures *only* how variables move together, nothing about their
individual shapes. Any joint distribution can be decomposed into (marginal
distributions) + (a copula), and built back up by sampling the copula and
transforming each column through its own marginal's inverse CDF (`ppf`).

`correlated_gbm.py` (see `correlated-assets.md`) already handles correlated
assets, but it implicitly uses a **Gaussian copula** on **lognormal
marginals** — both choices baked into the Cholesky-on-normal-shocks
construction. Reach for an explicit copula instead when either assumption
needs to be broken:

- **Marginals aren't Gaussian/lognormal.** E.g. combining `stable_distribution.py`'s
  alpha-stable return marginals with a correlation structure — `correlated_gbm.py`
  can't do this at all, since it only ever produces normal shocks.
- **The dependence itself needs to not be Gaussian.** The Gaussian copula
  has *zero* tail dependence — even at high linear correlation, the
  probability of a joint extreme event vanishes faster than the marginals
  alone would suggest. This was a real, widely-discussed contributor to
  underestimated joint-default risk in the 2008 financial crisis (the
  Gaussian copula wasn't the sole cause, but the zero-tail-dependence
  property is a real and well-documented mechanism, not a folk myth).
- **Real assets crash together more than they rally together** (or vice
  versa) — an asymmetric pattern neither the Gaussian nor the (symmetric)
  t-copula can represent, but an Archimedean copula like Clayton can.

If none of that applies — the marginals really are close to lognormal and
symmetric tail behavior is an acceptable simplification — `correlated_gbm.py`
is simpler and faster; don't reach for a copula by default.

## The three families implemented here

| Family | Tail dependence | Parameter | When to use |
|---|---|---|---|
| Gaussian | None (zero at any correlation) | Correlation matrix | Baseline / matches `correlated_gbm.py`'s implicit assumption |
| Student-t | Symmetric (upper = lower) | Correlation matrix + degrees of freedom | Assets that crash together AND rally together more than Gaussian predicts; lower df = fatter tail dependence |
| Clayton (Archimedean) | Asymmetric — strong lower, weak upper | Single alpha (exchangeable — all pairs share it) | Modeling correlated crashes specifically, e.g. "how likely is a joint drawdown across this portfolio" |

**Verified empirically** (reference script's `--demo`): at the *same*
Kendall's tau (0.6, so all three copulas agree on overall rank-correlation
strength), the empirical `P(asset 2 below its 5th percentile | asset 1
below its 5th percentile)` was 0.51 for Gaussian, 0.58 for a t-copula at
df=4, and 0.79 for Clayton — a large, real difference in joint-crash
probability hidden entirely inside a copula-family choice that a single
correlation number can't distinguish. This is the concrete case for not
defaulting to Gaussian dependence without thinking about it.

**Clayton's exchangeability is a real limitation**, not just a modeling
choice: the reference implementation's `clayton_copula_sample` uses one
shared `alpha` across every pair of assets, so it can't express "A and B
crash together strongly, but C is only weakly linked to either" the way
a full correlation matrix can. For non-exchangeable Archimedean structure,
nested/hierarchical Archimedean copulas exist in the literature but aren't
implemented here — treat Clayton as best-suited to a portfolio where
"the whole thing crashes together" is a reasonable simplification, not a
general n-asset dependence tool.

## Kendall's tau, not Pearson correlation, is the natural copula parameter

Pearson correlation is defined on the marginals' actual values and changes
under a nonlinear marginal transform — meaningless as a copula-only
concept. Kendall's tau is a rank-based measure of concordance that depends
only on the copula, not the marginals, which is why it's the right
parameter to reason about and specify:

- Gaussian/t (elliptical) copulas: `rho = sin(pi/2 * tau)` — same formula
  for both families, since tau depends only on the correlation matrix, not
  the t-copula's df.
- Clayton: `tau = alpha/(alpha+2)`, so `alpha = 2*tau/(1-tau)` — only valid
  for `0 < tau < 1` (Clayton in this parameterization has no
  negative-dependence range).

Both conversions are implemented and were used to tune the three families
to the *same* tau in the `--demo` comparison above — this is what makes the
tail-dependence gap a fair comparison rather than an artifact of using
different overall dependence strengths.

## Not implemented: Gumbel and Frank

The other two standard Archimedean copulas are documented, not
implemented, because their frailty-sampling requirements are genuinely
harder:

- **Gumbel** (upper-tail dependence, the mirror image of Clayton) needs
  its frailty variable sampled from a *positive* alpha-stable distribution
  — the same Chambers-Mallows-Stuck family as `stable_distribution.py`,
  but scipy has no built-in one-sided/positive-stable sampler, so this
  would need a custom implementation of the positive-stable special case.
- **Frank** (symmetric, no tail dependence but flexible mid-range
  dependence) needs frailty sampled from a Logarithmic(p) distribution,
  which scipy also doesn't provide a sampler for.

Both are real, addressable gaps if a future need calls for upper-tail-only
or Frank-style dependence specifically — not silently worked around here.

## Reviewed reference implementation

`../../monte-carlo-workspace/reference-scripts/copula_sampling.py`
— Gaussian, t, and Clayton copula samplers, Kendall's tau conversion
helpers, and `apply_marginals()` to combine any copula's uniform output
with arbitrary (and independently chosen per-asset) marginal `ppf`
functions. Verified: mixing a normal marginal with a lognormal marginal
under a shared Clayton dependence structure recovered both marginals'
theoretical mean/std and the target Kendall's tau simultaneously.

## Validation

- **Marginal uniformity check**: every copula sampler's raw output should
  be ~Uniform(0,1) per column before `apply_marginals` — a quick
  `.min()`/`.max()`/`.mean()` check catches a sampling-formula bug
  immediately (verified during development: Clayton's raw output ranged
  ~[2e-6, 0.9999994] with mean ~0.500, as expected).
- **Kendall's tau recovery**: sample, compute `scipy.stats.kendalltau`,
  compare to the target — cheap and catches both alpha-conversion errors
  and copula-formula errors.
- **Tail-dependence comparison across families at matched tau** (as done
  in `--demo`) is the real validation that a copula choice matters — don't
  skip this when justifying a Clayton-over-Gaussian choice to a decision
  that depends on it.
