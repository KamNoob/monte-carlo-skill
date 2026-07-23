# Particle Filters / Sequential Monte Carlo Reference

## What problem this solves, and how it differs from everything else in this skill

Every other reference in this skill either simulates forward from known
parameters (GBM, correlated GBM, copulas) or fits a distribution/process
to an observed series (stable, VG, EVT) or samples from a posterior over
static parameters (MCMC, HMC/NUTS). A particle filter solves a different
problem: **recovering a hidden state that evolves over time**, when the
observations are noisy and the state/observation relationship is
nonlinear or non-Gaussian enough that no closed-form filter exists. The
canonical finance example: a stock's volatility is not directly observed
— only noisy returns driven by that volatility are — and volatility
itself evolves over time (volatility clustering). A particle filter
recovers the posterior distribution of volatility at each point in time
given only the returns observed so far.

**If the model is linear-Gaussian, use a Kalman filter instead** — it
gives the exact posterior in closed form, with no Monte Carlo noise and
O(1) cost per step instead of O(N particles). This script's own
validation demo uses a Kalman filter as ground truth for exactly this
reason (see below) — it is the right tool when it applies, and the
particle filter's job in the demo is only to converge to what the Kalman
filter already gives exactly.

## The bootstrap filter (Sequential Importance Resampling)

Maintain N weighted particles representing the filtering posterior
`p(x_t | y_1:t)`. At each step:
1. **Propagate**: sample each particle forward through the transition
   model `x_t ~ p(x_t | x_{t-1})` — the "bootstrap" trick is proposing
   from the transition prior itself rather than a smarter importance
   density informed by the new observation, which is simple to implement
   but wastes particles when the transition is diffuse relative to how
   informative the observation is (a known limitation, not addressed
   here — see "what's not implemented" below).
2. **Reweight**: multiply each particle's weight by the observation
   likelihood `p(y_t | x_t)`.
3. **Resample** when the effective sample size (ESS) drops below a
   threshold (default N/2), to avoid weight degeneracy — a few particles
   accumulating all the weight while the rest become numerically
   worthless "zombie" particles that still cost compute but contribute
   nothing.

`systematic_resample` implements the standard O(N) systematic resampling
scheme (one random offset determines all N stratified draws via inverse-
CDF lookup) — lower variance than naive multinomial resampling (N
independent draws).

## Validation: converge to an exact Kalman filter, since most models have no other ground truth

Most nonlinear/non-Gaussian models a particle filter is actually used for
have no closed-form answer to check against — which makes validating a
new implementation harder than for, say, a distribution fit where you can
just check recovered moments. The standard trick: construct a
**linear-Gaussian** state-space model (where a Kalman filter gives the
exact answer), run the particle filter on the same data, and confirm the
particle filter's output converges to the Kalman filter's as N grows.

**Verified**: on a linear-Gaussian AR(1) model (`x_t = 0.9*x_{t-1} +
N(0,0.5^2)`, `y_t = x_t + N(0,0.3^2)`), the particle filter's mean
absolute error against the exact Kalman-filtered mean shrank as N grew —
0.022 at N=200, 0.007 at N=2000, 0.002 at N=20000 — consistent with the
expected O(1/sqrt(N)) Monte Carlo convergence rate.

**The marginal log-likelihood estimate also converges to the Kalman
filter's exact value** (computed independently via the prediction-error
decomposition, not derived from the particle filter). Note the **known
finite-N negative bias** in this
convergence (a Jensen's-inequality effect — the particle filter's
likelihood estimator is unbiased in the likelihood scale but the *log* of
an unbiased estimator is biased downward for finite N) — don't be
surprised if the PF's log-likelihood estimate at moderate N runs slightly
below the true value; that's expected, not a sign of a bug, and shrinks
as N grows.

## What's NOT implemented here

- **Only the bootstrap proposal** (transition-prior importance density).
  A fully general particle filter can use any importance density,
  including one informed by the current observation (auxiliary particle
  filters, or an extended/unscented-Kalman-filter-based proposal) — more
  efficient (fewer particles needed) but more complex to implement
  correctly. The bootstrap filter is the right starting point and is what
  most tutorials/papers mean by "particle filter" without qualification.
- **No parameter estimation.** This filters a KNOWN state-space model
  (fixed transition/observation parameters) — it does not estimate those
  parameters from data. Particle MCMC (PMMH — particle marginal
  Metropolis-Hastings) combines this filter's log-likelihood estimate
  with an MCMC sampler over the static parameters for that purpose; not
  implemented here, but the log-likelihood output this filter already
  computes is exactly the ingredient PMMH needs.
- **No smoothing.** This computes the *filtering* distribution
  `p(x_t | y_1:t)` (using only observations up to time t), not the
  *smoothing* distribution `p(x_t | y_1:T)` (using the whole series) —
  smoothing needs a separate backward pass.

## Bugs found and fixed by adversarial review

An adversarial review found and this was fixed before the implementation
was considered done:

- **The main loop unconditionally transitioned particles before weighting
  them against the first observation**, contradicting the documented
  contract that `x0_sampler` draws directly from `p(x_0)` (the classic
  bootstrap filter, per Gordon/Salmond/Smith 1993, weights the initial
  particles against `observations[0]` with no transition step first).
  The reviewer constructed a concrete near-degenerate test case (prior
  concentrated at `x0=5.0`, single observation `y0≈5.038`) proving a real
  ~2% systematic bias, not just a naming quibble — fixed by skipping the
  transition step on the first iteration; verified the fix recovers the
  theoretically expected filtered mean (5.00000) exactly on the same test
  case, and the Kalman-convergence demo's data-generating process and
  oracle were updated to match the corrected convention (both now treat
  `x[0]` as the `p(x_0)` draw itself, not one transition past it).
- **`np.isfinite`'s NaN/-inf conflation** in the log-likelihood validity
  check would hard-fail on legitimate `-inf` log-likelihoods (a normal
  occurrence for observation models with bounded support — a particle
  outside the support should get zero likelihood, not trigger an error).
  Fixed to reject only genuine `NaN` (always a bug) while allowing `-inf`
  (a valid degenerate weight), with a separate check for total weight
  collapse (every particle `-inf`) since that case genuinely has no
  valid particles left to filter with.

## Reviewed reference implementation

`../../monte-carlo-workspace/reference-scripts/particle_filter.py`
— `bootstrap_particle_filter`, `systematic_resample`,
`effective_sample_size`, and a private Kalman filter used only as the
validation oracle. The `--demo` reproduces the Kalman-convergence check
above at three particle counts.

## Validation checklist for a new state-space model

- **Linear-Gaussian sanity check first**, even if the real model isn't —
  a bug in the generic filter machinery (resampling, weight updates) will
  usually show up here before it shows up in a harder nonlinear case.
- **ESS trace**: should mostly stay well above the resample threshold
  between resamples; an ESS that collapses to ~1 immediately after every
  resample is a sign the transition/observation models are badly
  mismatched to the data (proposals essentially never land near where the
  likelihood is high).
- **Particle-count convergence**: run at increasing N and confirm the
  filtered estimates stabilize — if increasing N doesn't visibly tighten
  the estimate, something in the model (not just Monte Carlo noise) is
  likely wrong.
- **Log-likelihood as a diagnostic**: report it whenever comparing
  competing state-space model specifications on the same data (e.g.
  different transition dynamics) — higher (less negative) is a better
  fit, same use as any other likelihood-based model comparison.
