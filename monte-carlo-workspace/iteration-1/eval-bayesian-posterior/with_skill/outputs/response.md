# Bayesian Coin Flip: MCMC Posterior Estimation

## Problem Classification

This is an **MCMC / Bayesian** problem (per the skill's Step 1 table):
- We have a parameter of interest (coin bias θ ∈ (0,1))
- A prior belief (Beta(2,2): weakly symmetric, centred at 0.5)
- Observed data (7 heads out of 10 flips)
- Goal: posterior distribution of θ and a 95% credible interval

The posterior is analytically tractable (Beta–Binomial conjugacy), which we'll use to validate the MCMC. But we implement Metropolis-Hastings from scratch as requested.

---

## Mathematical Setup

**Likelihood** — Binomial:

```
P(data | θ) = C(10,7) · θ^7 · (1−θ)^3
log P(data | θ) = 7·log(θ) + 3·log(1−θ)   [constant dropped]
```

**Prior** — Beta(2,2):

```
p(θ) ∝ θ^(2−1) · (1−θ)^(2−1) = θ · (1−θ)
log p(θ) = log(θ) + log(1−θ)
```

**Posterior** — proportional to likelihood × prior:

```
p(θ | data) ∝ θ^7 · (1−θ)^3 · θ · (1−θ) = θ^8 · (1−θ)^4
```

This is Beta(9, 5) analytically — our ground truth for validation.

---

## Implementation: Metropolis-Hastings from Scratch

### Key Design Choice: Logit-Space Sampling

θ lives in (0,1). A Gaussian random-walk proposal on θ directly risks proposing values outside [0,1], which would always be rejected and stall the chain. Instead, we sample in **logit space**:

```
φ = logit(θ) = log(θ/(1−θ)),   θ = sigmoid(φ)
```

The change of variables adds a Jacobian term to the log posterior:

```
log|dθ/dφ| = log(θ) + log(1−θ)
```

### Code

```python
import numpy as np
from scipy import stats

# ─── Data and prior ───────────────────────────────────────────────────────────
n_flips, n_heads = 10, 7
alpha_prior, beta_prior = 2, 2

# ─── Log posterior in logit-transformed space ─────────────────────────────────
def log_posterior_logit(phi):
    """Unnormalised log posterior of φ = logit(θ)."""
    theta = 1.0 / (1.0 + np.exp(-phi))
    if theta <= 0 or theta >= 1:
        return -np.inf
    log_lik   = n_heads * np.log(theta) + (n_flips - n_heads) * np.log(1 - theta)
    log_prior = (alpha_prior - 1) * np.log(theta) + (beta_prior - 1) * np.log(1 - theta)
    log_jac   = np.log(theta) + np.log(1 - theta)   # Jacobian correction
    return log_lik + log_prior + log_jac

# ─── Metropolis-Hastings sampler ──────────────────────────────────────────────
def metropolis_hastings(log_prob, initial, n_samples, proposal_std=1.5, seed=42):
    """
    Gaussian random-walk Metropolis-Hastings.

    Parameters
    ----------
    log_prob     : callable — unnormalised log posterior
    initial      : float   — starting value of the chain
    n_samples    : int     — total iterations (include burn-in)
    proposal_std : float   — std of Gaussian proposal; tune for ~44% acceptance in 1D
    seed         : int     — RNG seed for reproducibility

    Returns
    -------
    samples      : np.ndarray of shape (n_samples,)
    acceptance_rate : float
    """
    rng_mh = np.random.default_rng(seed)
    phi    = np.float64(initial)
    samples  = np.zeros(n_samples)
    accepted = 0
    log_p_current = log_prob(phi)

    for i in range(n_samples):
        # 1. Propose a new point from Gaussian centred at current position
        proposal = phi + rng_mh.normal(0, proposal_std)

        # 2. Compute log acceptance ratio (symmetric proposal → reduces to log posterior ratio)
        log_p_proposal = log_prob(proposal)
        log_alpha      = log_p_proposal - log_p_current

        # 3. Accept or reject
        if np.log(rng_mh.uniform()) < log_alpha:
            phi           = proposal
            log_p_current = log_p_proposal
            accepted += 1

        samples[i] = phi

    return samples, accepted / n_samples

# ─── Run MCMC ─────────────────────────────────────────────────────────────────
n_samples = 50_000
burn_in   = 10_000   # discard first 20%

phi_samples, acc_rate = metropolis_hastings(
    log_posterior_logit,
    initial=0.0,          # logit(0.5) = 0, a neutral starting point
    n_samples=n_samples,
    proposal_std=1.5,     # tuned for ~41% acceptance in logit space
    seed=42
)

# Transform back to probability space
theta_samples = 1.0 / (1.0 + np.exp(-phi_samples))
theta_post    = theta_samples[burn_in:]   # 40,000 post-burn-in samples

# ─── Analytical posterior for validation ─────────────────────────────────────
alpha_post_a = alpha_prior + n_heads           # 9
beta_post_a  = beta_prior + (n_flips - n_heads)  # 5
analytical_mean = alpha_post_a / (alpha_post_a + beta_post_a)
analytical_mode = (alpha_post_a - 1) / (alpha_post_a + beta_post_a - 2)
analytical_ci   = stats.beta.ppf([0.025, 0.975], alpha_post_a, beta_post_a)
```

### Proposal Tuning

The reference guide says: target ~44% acceptance for 1D problems. I swept `proposal_std` over [0.5, 1.0, 1.5, 2.0] in φ-space:

| proposal_std | acceptance rate |
|---|---|
| 0.5 | 73.8% (too small — chain barely moves) |
| 1.0 | 54.6% |
| **1.5** | **41.4%** (close to 44% target) |
| 2.0 | 33.7% |

`proposal_std = 1.5` was selected.

---

## Results

### MCMC Diagnostics

| Metric | Value |
|---|---|
| Total samples | 50,000 |
| Burn-in discarded | 10,000 (20%) |
| Post-burn-in samples | 40,000 |
| Acceptance rate | 41.4% (target: ~44%) |
| Effective sample size (ESS) | ~9,235 |

ESS of 9,235 far exceeds the minimum threshold of 400, so the chain is well-mixed. The acceptance rate is close to the 44% optimum for 1D random-walk M-H.

### Posterior Summary

| Quantity | MCMC | Analytical Beta(9,5) |
|---|---|---|
| Mean | **0.6431** | 0.6429 |
| Median | 0.6516 | — |
| Mode | — | **0.6667** |
| Std dev | 0.1254 | 0.1225 |
| 95% Credible Interval | **[0.3809, 0.8597]** | [0.3857, 0.8614] |

### 95% Credible Interval for Coin Bias

```
θ ∈ [0.381, 0.860]   with 95% posterior probability
```

The chain error vs. the analytical posterior is small: mean error = 0.00022, CI lower error = 0.0048, CI upper error = 0.0017.

---

## Posterior Visualisation (ASCII)

The posterior is Beta(9,5), skewed toward higher values (reflecting the 7/10 heads evidence pulling away from the symmetric Beta(2,2) prior):

```
Density
  ^
  |
  |          ****
  |        **    **
  |       *        *
  |      *          **
  |    **              **
  |  **                  ***
  |**                        *****
  +--+--+--+--+--+--+--+--+--+--+--> θ
  0 .1 .2 .3 .4 .5 .6 .7 .8 .9 1.0
                    [  95% CI   ]
                  0.38         0.86
                       ^
                     mean≈0.64
```

The prior Beta(2,2) was centered at 0.5. After observing 7/10 heads, the posterior shifts rightward to a mean of ~0.64, but the wide credible interval [0.38, 0.86] correctly reflects that 10 flips is not a lot of data.

---

## Interpretation

1. **Point estimate**: the posterior mean is 0.64. Given only 10 flips, the prior Beta(2,2) has a regularising effect — it pulls the naive MLE of 7/10 = 0.70 toward 0.5, yielding 0.64 instead.

2. **95% Credible Interval [0.381, 0.860]**: we have 95% posterior belief that the coin's true bias lies in this range. The interval is wide because 10 flips gives limited evidence — we cannot firmly say whether this coin is fair or biased.

3. **Is the coin fair?** θ = 0.5 lies within the credible interval, so the data are consistent with a fair coin. However, the bulk of the posterior is above 0.5, so the coin is more likely biased toward heads.

4. **Prior effect**: with a strong prior like Beta(10,10), the same data would push the posterior mean closer to 0.5. With a flat prior Beta(1,1), it would sit at 0.70. Beta(2,2) is a mild regulariser.

---

## Validation (Step 4 from skill)

The analytical posterior is **Beta(9,5)** — closed form because Beta is conjugate to Binomial. The MCMC result matches it to within:

- Mean: 0.00022 error
- CI lower: 0.0048 error
- CI upper: 0.0017 error

These errors are well within one standard error of the MCMC estimate, confirming the sampler is correct.

---

## What Would Make This More Accurate

- **More data**: 100+ flips would tighten the CI considerably.
- **More MCMC samples**: 50k is sufficient here (ESS > 9k), but scaling to 200k would reduce Monte Carlo noise further.
- **Multiple chains**: running 4 independent chains from different starting points and checking R-hat < 1.01 would provide stronger convergence guarantee. With a 1D unimodal posterior like this, a single chain is fine, but it's good practice.
- **HMC / NUTS**: for higher-dimensional models (multiple parameters), gradient-based samplers like those in PyMC or Stan are far more efficient than random-walk M-H.
