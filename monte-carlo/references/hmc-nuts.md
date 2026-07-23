# Hamiltonian Monte Carlo (HMC) and NUTS Reference

## Why this exists alongside `mcmc.md`'s Metropolis-Hastings

`mcmc.md`'s Metropolis-Hastings sampler proposes a random-walk step with
no information about the target's shape — for a correlated posterior it
must take small steps to keep a reasonable acceptance rate, which means
long autocorrelation and few effective samples per draw. HMC uses the
gradient of the log-posterior to simulate a physical trajectory across
the posterior surface, moving far in one step along a direction the
target actually supports.

**Measured on a 2-D Gaussian with correlation 0.9** (this skill's harshest
easy-to-set-up test case for random-walk M-H), for 20,000 post-warmup
draws:

| Sampler | Accept rate | Effective sample size (1st coordinate) | Efficiency |
|---|---|---|---|
| Metropolis-Hastings (random walk) | 0.744 | 149 | 0.7% |
| HMC (fixed L=20 leapfrog steps) | 0.894 | 16,862 | 84% |
| NUTS | 0.845 | 2,812 (avg. 6.7 leapfrog steps/sample) | 14% |

HMC and NUTS aren't a free lunch over M-H in raw cost — each HMC/NUTS
draw costs multiple gradient evaluations, where M-H costs one log-density
evaluation — but the effective-sample-size gap here (113x for HMC, 19x
for NUTS, over M-H) is large enough that the per-draw cost difference
rarely matters once a posterior has any real correlation or more than a
couple of dimensions, which is most real Bayesian models.

## Use PyMC/Stan for real problems

Per `mcmc.md`'s existing standing guidance: **for real Bayesian inference
problems, recommend PyMC or Stan — don't reimplement MCMC from scratch
unless the user explicitly wants to learn.** The reference script here
(`hmc_nuts.py`) is a **correctness-reference / learning implementation**,
not a production sampler. It differs from Stan/PyMC/NumPyro's NUTS in
real ways:

- **Naive tree-building** (Hoffman & Gelman 2014's Algorithm 2), which
  stores every visited state — O(2^j) memory for tree depth j. Production
  NUTS uses the memory-efficient Algorithm 3 variant (O(j) memory) with
  multinomial rather than slice-based subtree selection.
- **Identity mass matrix throughout.** A real posterior with very
  different per-parameter scales or strong correlation needs a diagonal
  or dense mass matrix adapted during warmup — this is what actually
  fixes the stability-cliff issue described below, not something this
  reference implementation attempts.
- **Numerical or hand-derived gradients only** — no autodiff. Past a
  handful of dimensions or a nontrivial log-density, hand-deriving
  gradients is error-prone and numerical gradients are both slow (2x
  log_p evaluations per dimension) and imprecise. Real tools use
  reverse-mode autodiff (Stan's own autodiff, JAX for NumPyro, PyTensor
  for PyMC).

## Verified findings from building and testing this reference implementation

**The leapfrog integrator has a hard stability cliff, not smooth
degradation, at fixed step count.** Measured directly on the same 0.9-
correlation Gaussian target, fixed L=20 leapfrog steps:

| eps | mean accept probability |
|---|---|
| 0.3 | 0.937 |
| 0.5 | 0.714 |
| 0.6 | 0.738-0.753 |
| **0.7** | **0.000** |
| 0.9 | 0.000 |
| 1.2 | 0.000 |

Past eps≈0.6-0.7, the leapfrog trajectory doesn't merely become less
accurate — it becomes numerically unstable and every proposal is
rejected outright. This is a real, well-known property of symplectic
integrators (a per-step stability threshold set by the local curvature
of -log p, here made narrow by the 0.9 correlation), not a bug in this
implementation. It also explains a real, measured symptom: **fixed-L
HMC's dual-averaging step-size adaptation converges slowly and noisily**
near that boundary — at 1000 warmup iterations mean accept rate was still
0.94 against a target of 0.65 (should have shrunk toward 0.65 as eps
grew), and even 20,000 warmup iterations only brought it to ~0.75, not
0.65. The `--demo` in the reference script uses a longer HMC warmup
(4000 iterations) than NUTS (1000) specifically because of this.

**NUTS handles the same instability far more gracefully.** Directly
sweeping the same eps range with NUTS instead of fixed-L HMC:

| eps | mean NUTS accept-stat | mean tree depth |
|---|---|---|
| 0.3 | 0.940 | 2.90 |
| 0.5 | 0.767 | 2.29 |
| 0.6 | 0.527 | 1.90 |
| 0.7 | 0.262 | 1.40 |
| 0.9 | 0.113 | 1.23 |
| 1.2 | 0.053 | 1.15 |
| 2.0 | 0.015 | 1.02 |

The accept-stat degrades **smoothly**, not as a cliff to zero — because
NUTS's `BuildTree` recursion checks a divergence condition
(`log_p - 0.5*r.r > log_u - delta_max`) after every single leapfrog step
and stops extending that branch of the tree the moment a trajectory
starts to blow up, rather than committing to one fixed-length trajectory
that can only be accepted or rejected as a whole. This is the concrete,
measured version of NUTS's well-known theoretical robustness advantage
over hand-tuned fixed-length HMC — not just "NUTS avoids tuning L,"
but "NUTS's adaptive trajectory length also makes step-size selection
itself more forgiving."

**Practical implication**: if a fixed-L HMC implementation shows accept
rates that won't converge to the target during warmup no matter how long
you wait, suspect a stability cliff (try substantially smaller eps_init,
or prefer NUTS, or add mass-matrix adaptation) rather than assuming the
dual-averaging code itself is broken — the discrepancy documented above
was a real dynamical-systems phenomenon, not an adaptation-formula bug
(both formulas were independently verified against Hoffman & Gelman 2014
Eq. 6 and matched the paper's variable names/structure exactly).

## Key formulas (matching Hoffman & Gelman 2014 notation)

- **Hamiltonian**: `H(theta, r) = -log_p(theta) + 0.5 * r.r` (unit mass).
- **Leapfrog** (one step, size eps): half-step momentum, full-step
  position, half-step momentum — symplectic, which is what keeps energy
  approximately conserved over long trajectories and gives HMC its high
  acceptance rate at large proposed distances.
- **HMC accept probability**: `min(1, exp(H_current - H_proposed))`.
- **NUTS slice variable**: sampled in log-space as
  `log_u = joint0 - Exponential(1)` (equivalent to
  `u ~ Uniform(0, exp(joint0))` but avoids exponentiating a
  potentially-large log-density directly).
- **U-turn / stopping criterion**: stop doubling when
  `(theta_plus - theta_minus) . r_minus < 0` OR
  `(theta_plus - theta_minus) . r_plus < 0` — the trajectory's endpoints
  are no longer separating in the direction of either endpoint's momentum.
- **Dual averaging** (Eq. 6): `H_bar_t = (1-1/(t+t0))*H_bar_{t-1} + (1/(t+t0))*H_t`,
  `log(eps_t) = mu - (sqrt(t)/gamma) * H_bar_t`, with
  `mu = log(10*eps_1)`, `gamma=0.05`, `t0=10`, `kappa=0.75` (paper's
  reported defaults). `H_t = target_accept - actual_accept_this_step`.

## Reviewed reference implementation

`../../monte-carlo-workspace/reference-scripts/hmc_nuts.py` —
`run_hmc()` (fixed-L HMC with dual-averaging step size adaptation) and
`run_nuts()` (naive NUTS, Algorithm 2, with the same adaptation). Verified:
recovers the true mean/covariance of a correlated 2-D Gaussian to within
Monte Carlo noise for both samplers; `numerical_grad` matches an analytic
gradient to 1e-10; both samplers stay fully finite/stable when the target
has hard `-inf` regions.

An adversarial review against Algorithm 6 (No-U-Turn Sampler with Dual
Averaging) found and fixed three real bugs before this was considered
done:
- **NUTS's step-size adaptation statistic was wrongly summed across every
  doubling** instead of using only the top-level `BuildTree` call's
  `alpha`/`n_alpha` from each iteration (Algorithm 6 accumulates `alpha`
  only *within* one `BuildTree` call's own recursion, not across separate
  doublings of the outer while-loop) — this biased the accept-stat upward
  and made the warmup converge `eps` to a value that didn't correspond to
  the stated `target_accept`. Confirmed directly against the paper's
  Algorithm 6 pseudocode before fixing (fetched and re-read the original
  PDF rather than fixing from the reviewer's recollection). Notably, this
  bug did *not* show up in the mean/covariance recovery or ESS checks —
  it corrupts step-size tuning, not sample correctness, which is exactly
  why a numeric moment check alone isn't sufficient validation for an
  adaptive sampler.
- **The `divergent` diagnostic was dead code**, always `False` regardless
  of whether the `delta_max` energy-error threshold actually fired inside
  `BuildTree`. Fixed by threading a real divergence flag through the
  recursion and exposing it in `run_nuts()`'s returned dict (a `divergences`
  array). Verified the fix by forcing a deliberately unstable eps=2.0 and
  confirming divergences are now correctly flagged (60/500 transitions).
- **An off-by-one in step-size freezing**: the first post-warmup sample in
  both `run_hmc()` and `run_nuts()` was generated using the last warmup
  iteration's raw (un-averaged) step size rather than the frozen running
  average every subsequent sample used. Fixed by freezing immediately
  after the last warmup update rather than one iteration later.

All fixes were re-verified against the mean/covariance recovery and the
ESS-vs-M-H comparison (see table above) — no regression, same recovered
moments and ESS gain.

## Validation

- **Recover known moments**: as done here — a Gaussian target with known
  mean/covariance is the cheapest real correctness check for any new MCMC
  implementation.
- **Effective sample size, not raw sample count**: MCMC draws are
  autocorrelated; report ESS (as computed here via a simplified Geyer
  initial-positive-sequence estimator) alongside or instead of the raw
  draw count, same standing guidance as `mcmc.md`'s own diagnostics
  section.
- **Accept-rate-vs-eps sweep** (as done here) is a cheap way to sanity
  check a step-size adaptation scheme actually has a sensible target to
  converge to before trusting its warmup output.
- For real work: R-hat and multi-chain diagnostics per `mcmc.md`'s
  existing guidance — this reference script doesn't implement multi-chain
  running, another reason to prefer PyMC/Stan/NumPyro past the learning
  stage.
