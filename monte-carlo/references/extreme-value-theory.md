# Extreme Value Theory (GEV / GPD) Reference

## When to reach for EVT instead of a full-distribution fit

`stable-distributions.md` and `variance-gamma.md` fit a distribution to
the *whole* returns series and read tail percentiles off that fit. EVT
instead fits **only the tail**, via the Pickands-Balkema-de Haan theorem:
for a high enough threshold, exceedances are asymptotically Generalized
Pareto distributed *regardless of the parent distribution* — the same GPD
shape applies whether the true underlying returns are Gaussian, stable,
or VG. This makes EVT the right tool specifically for deep-tail questions
("1-in-1000-day" VaR/CVaR, 99.9%+ confidence) where the bulk of the data
can't inform the answer and a full-distribution fit is extrapolating on
faith. It's the wrong tool for everyday 95-99% VaR, where a full-
distribution fit uses the data more efficiently (POT discards everything
below the threshold) and doesn't need the threshold-selection judgment
call EVT requires.

## Two approaches, this skill implements the POT/GPD one

- **GEV (block maxima)**: fit to one value per block (e.g. worst day per
  month). Answers "what's the distribution of the single worst
  observation in a period this long."
- **GPD (peaks-over-threshold, POT)**: fit to every exceedance over a
  threshold. Answers "given we're already past threshold u, how much
  further could it go" — uses far more of the tail data than block maxima
  for the same history, which is why `extreme_value.py` implements this
  path for VaR/CVaR. `fit_gev` is included too, for the block-maxima
  question specifically when that's what's being asked.

## Critical scipy gotcha — verified empirically, not just from docs

scipy's two extreme-value distributions use **opposite shape-parameter
sign conventions relative to each other**:

- `scipy.stats.genpareto`'s `c` matches the standard EVT shape `xi`
  directly — confirmed by fitting exceedances generated with a known
  `xi=0.25` and recovering `c=0.248` (no sign flip).
- `scipy.stats.genextreme`'s `c` is the **negative** of the standard
  `xi` — scipy's own docs flag this ("several sources... use the
  opposite convention"), and it was confirmed here directly: checking
  which sign of `c` produces an unbounded-above support (the correct
  behavior for `xi>0`, heavy Fréchet tails) showed `c=-0.25` was
  unbounded above and `c=+0.25` was bounded above — i.e.
  `scipy_genextreme_c == -xi_standard`.

This matters because the Pickands-Balkema-de Haan theorem says the GEV
shape and the GPD shape for the *same* underlying tail should be the
*same* `xi` — a natural thing to want to cross-check, and exactly where a
silent sign flip would produce a confidently-wrong answer. `fit_gev` in
the reference script converts scipy's raw `c` to standard `xi`
internally (`xi = -c`) specifically so this comparison is safe by
default rather than a trap waiting for whoever uses both functions
together.

## VaR/CVaR formulas — derived and verified, not recited

Derived from the POT tail-probability approximation
`P(X > u+y) = (Nu/n) * (1 + xi*y/sigma)^(-1/xi)` (solve for the `y` where
this equals the target tail probability) and, for CVaR, the GPD's
**threshold-stability property** (exceedances of a GPD over any higher
threshold are themselves GPD with the same `xi` and a shifted `sigma`,
so the tail beyond VaR has a known closed-form mean):

```
tail_prob = (1-confidence) * n / Nu
VaR   = u + (sigma/xi) * (tail_prob^(-xi) - 1)          [xi != 0]
VaR   = u - sigma * log(tail_prob)                       [xi == 0]
CVaR  = VaR/(1-xi) + (sigma - xi*u)/(1-xi)                [xi < 1; undefined/infinite for xi>=1]
```

**Verified end-to-end** against ground truth: simulated 500,000 points
from a *known* Student-t(df=4) distribution, fit GPD via POT at a 95th-
percentile threshold, and compared the GPD-derived VaR/CVaR at 99.9%
confidence to the *true* VaR/CVaR computed directly from the known
t-distribution (not from another fit). Result: VaR within 0.57%, CVaR
within 1.38% — extrapolating correctly far beyond the fitting threshold.

**CVaR is more sensitive to xi-estimation noise than VaR** — verified
directly: with a well-converged `xi` (large sample, deep threshold), both
matched true values to within 0.1%; with less converged `xi` the VaR
error stayed small but the CVaR error grew to ~2%, because CVaR's formula
divides by `(1-xi)` directly while VaR's dependence on `xi` is gentler.
Don't be surprised if a GPD-derived CVaR looks noisier than the VaR from
the same fit — that's expected, not a red flag on its own.

**xi is a genuinely high-variance parameter to estimate from tail data —
don't trust a point estimate alone.** An adversarial review's 300-trial
simulation (true xi=0.25) found: at 25 exceedances, fitted xi had
std~0.28 (the estimated *sign* — bounded vs. unbounded tail — flipped
routinely); at 500 exceedances, std was still ~0.05 with visible bias.
`fit_gpd` now (a) requires at least 100 exceedances as a hard floor
against outright degenerate fits — not a stability guarantee — and (b)
bootstraps a 95% CI on `xi`/`sigma` (resampling exceedances with
replacement and refitting, default 200 replicates), which `gpd_var_cvar`
propagates into a CI on VaR/CVaR too. Always check `xi_ci_95` and
`var_ci_95`/`cvar_ci_95` before trusting a point estimate — a wide CI
means "more data needed," not "the code is wrong."

## Threshold selection: a real judgment call, not something to automate away

Too low a threshold violates the POT theorem's asymptotic assumption
(bias — non-tail data pollutes the fit). Too high leaves too few
exceedances (variance — an unstable fit). `choose_threshold_percentile`
in the reference script picks a fixed percentile (default 95th) as a
starting point, not a substitute for the standard EVT diagnostic (a
mean-residual-life plot: mean exceedance vs. threshold, looking for where
it flattens into roughly a straight line) which requires visual
judgment. At minimum, do what the reference script's demo does: refit at
a couple of nearby thresholds and check `xi` is reasonably stable — a
large swing in `xi` between adjacent thresholds is a sign the fit isn't
trustworthy yet.

## Reviewed reference implementation

`../../monte-carlo-workspace/reference-scripts/extreme_value.py`
— `fit_gev` (sign-corrected), `fit_gpd`, `gpd_var_cvar`,
`choose_threshold_percentile`. The `--demo` reproduces the ground-truth
Student-t validation described above.

## Validation

- **Ground-truth check against a known distribution** (as done here) is
  the strongest available validation for any new EVT code — most
  synthetic/real financial data doesn't have a "true" answer to check
  against, so do this once against a known distribution before trusting
  the implementation on real data.
- **Threshold stability**: refit at 2-3 nearby thresholds, check `xi`
  doesn't swing wildly.
- **GEV/GPD cross-check**: if both a block-maxima GEV fit and a POT GPD
  fit are done on the same underlying series, their `xi` values (using
  this script's sign-corrected convention for both) should be in the
  same ballpark — a large disagreement is a signal something's off with
  one of the fits or the block/threshold choice, not two independently
  "correct" different answers.
- Sample-size floor: the reference script requires at least 100
  exceedances/blocks, which is itself a minimum, not a target — EVT
  parameter estimates (especially `xi`) are notoriously high-variance
  with small tail samples (verified: sign-flipping instability at 25
  exceedances); more is better whenever available, and the bootstrap CI
  on `xi`/VaR/CVaR (see above) is the honest way to see how much better
  is actually needed for a given dataset.
