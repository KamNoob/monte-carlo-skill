# Variance-Gamma (VG) Reference

## Where this sits between GBM and alpha-stable

`references/financial.md`'s GBM assumes Gaussian log-returns (no skew, no
excess kurtosis by construction). `references/stable-distributions.md`'s
alpha-stable family goes to the other extreme: power-law tails, possibly
infinite variance. Variance-Gamma sits between them: **finite moments of
every order**, but skewness and excess kurtosis that are independently
controllable and estimated from data rather than assumed away. It's the
standard "semi-heavy-tailed" alternative when the tails clearly aren't
Gaussian but a claim of infinite variance would be too strong. It
underlies the Madan-Carr-Chang (1998) option pricing model and traces to
Madan & Seneta (1990).

Use VG when:
- Returns show real skewness/excess kurtosis but nothing as extreme as
  what a stable fit would imply (check: does an alpha-stable fit on the
  same data land close to alpha=2? If so, VG is the better-fitting,
  simpler, finite-variance model)
- A closed-form, fast-to-simulate alternative to GBM is wanted without
  the slow/fragile fitting behavior documented in `stable-distributions.md`
  (VG's method-of-moments fit here runs in under a second vs. stable's
  measured 70+ seconds)

## Construction: a Gamma-subordinated Brownian motion

`X_t = mu*t + theta*G_t + sigma*W(G_t)`, where `G_t` is a Gamma process
(`G_t ~ Gamma(shape=t/nu, scale=nu)`, so `E[G_t]=t`, `Var[G_t]=t*nu`) and
`W` is an independent standard Brownian motion. Intuition: time itself is
randomized by an independent Gamma clock, and returns accumulate
Brownian-motion noise at that randomized pace — periods where the clock
runs fast look like high-volatility regimes, producing fatter tails and
(via `theta`) asymmetry than a fixed-pace Brownian motion (GBM) can.

- `sigma`: base volatility scale
- `nu`: controls excess kurtosis (`nu=0` degenerates exactly to Gaussian/GBM)
- `theta`: controls skew (`theta<0` gives left skew, the typical
  asymmetry seen in equity returns)

## Why NOT `scipy.stats.genhyperbolic`

VG is a subclass of the generalized hyperbolic (GH) family, and scipy
does implement the general GH distribution — but VG arises as a
*boundary/limiting* case of GH (`delta -> 0`), not a generic interior
point. scipy's own docs warn: *"For distributions that are a special case
[of genhyperbolic] such as Student's t, it is not recommended to rely on
the implementation of genhyperbolic... the methods of the specific
distributions should be used"* — the same caveat applies to VG. This
skill implements VG directly via its own Gamma-subordination construction
instead, which needs no Bessel functions for sampling and has closed-form
moments — avoiding scipy's documented boundary-case fragility entirely
rather than working around it after the fact.

## Moment formulas — derived and verified, not recited

The moment formulas used for fitting were derived from VG's cumulant
generating function `K(u) = -(t/nu)*log(1 - nu*theta*u - 0.5*nu*sigma^2*u^2)`
(a standard result: MGF of a normal variance-mean mixture with a Gamma
subordinator, via the tower property and the Gamma distribution's MGF)
via Taylor expansion, then **verified against 3-million-path simulation**
before being trusted (mean/variance/skewness/excess-kurtosis all matched
simulated values closely) — not taken from an unverified formula in the
literature, per this skill's standing fact-check practice.

```
mean          = t * theta
var           = t * (sigma^2 + nu*theta^2)
skew          = nu*theta*(3*sigma^2 + 2*nu*theta^2) / (sqrt(t) * (sigma^2+nu*theta^2)^1.5)
excess_kurt   = 3*nu*(sigma^4 + 4*nu*theta^2*sigma^2 + 2*nu^2*theta^4) / (t * (sigma^2+nu*theta^2)^2)
```

Sanity check: when `theta=0` (symmetric case), `excess_kurt = 3*nu/t` —
matches the commonly-cited symmetric-VG kurtosis formula, a useful
cross-check that the fuller derivation above is consistent with it.

## Fitting: method-of-moments, and a real instability near the Gaussian boundary

`fit_vg_moments` matches sample (variance, skewness, excess kurtosis) to
the formulas above via a 2-D nonlinear solve for `(nu, theta)` — `sigma`
is eliminated via the variance identity, `mu` recovered from the sample
mean afterward. This is simpler and more numerically robust than MLE via
VG's closed-form (Bessel-function) density, deliberately avoiding the
same kind of boundary-case fragility scipy warns about for `genhyperbolic`
— at some cost in statistical efficiency versus full MLE.

**Verified, real instability**: near the Gaussian boundary (small sample
excess kurtosis), the method-of-moments system is only weakly identified.
Concretely tested: a 5000-point sample drawn from an *exactly Gaussian*
population had sample skew=0.055, excess kurtosis=0.089 (both near zero,
as expected purely from finite-sample noise) — but fitting VG to those
exact moments gives `nu=0.029, theta=0.64`, a mathematically exact root
(residual ~1e-14, confirmed identical across 25+ different solver
starting points — not a solver artifact). The mechanism: small nu forces
theta to be disproportionately large to reproduce even a small realized
skew, since theta enters the skew formula divided by nu. `fit_vg_moments`
flags this automatically (`nu < 0.05` triggers a warning) — **always
check for this warning before trusting a fit**, and consider whether a
plain Gaussian model already explains the data adequately when it fires.

**The opposite extreme is also unstable, just less obviously.** As true
`nu` grows, VG's higher moments get noisier in finite samples — verified
by fitting synthetic data across a range of true `nu`: recovery is good
up to `nu~1`, visibly biased (`sigma` undershoots) by `nu~3`, and the
solve fails outright (correctly raising an error rather than returning a
silently bad fit) by `nu~15`. The degrading-but-still-succeeding region in
between (`nu~3-8`) is flagged with a second warning (`nu > 3`) — don't
assume "the solve converged" means "the fit is trustworthy" at large `nu`
any more than it does near the Gaussian boundary.

Also note: `fit_vg_moments` uses bias-corrected (`bias=False`) sample
skewness/kurtosis to match the unbiased (`ddof=1`) sample variance already
in use — mixing biased and unbiased moment estimators in the same
nonlinear solve would introduce a small, avoidable inconsistency that
compounds exactly in the fragile small-sample regimes described above.

## Reviewed reference implementation

`../../monte-carlo-workspace/reference-scripts/variance_gamma.py`
— `vg_cumulants`, `simulate_vg`, `fit_vg_moments` (with the boundary-
instability warning above), and `vg_vs_normal_var_cvar` (same
Gaussian-comparison pattern as `stable_distribution.py`).

## Validation

- **Moment-matching round-trip**: after fitting, recompute `vg_cumulants`
  from the fitted parameters and confirm they reproduce the sample
  moments that were fit to (a basic self-consistency check, not proof of
  a good fit to the actual data shape — VG only matches 4 moments, not
  the full distribution).
- **Watch for the `nu < 0.05` warning** — see above.
- **Compare to a stable-distribution fit on the same data** (see
  `stable-distributions.md`) — if the stable fit's alpha lands close to 2,
  the extra complexity of allowing infinite variance isn't earning its
  keep and VG (or even plain Gaussian) is the better-justified model.
- Standard Gaussian-comparison VaR/CVaR validation as in
  `stable-distributions.md`'s validation section.
