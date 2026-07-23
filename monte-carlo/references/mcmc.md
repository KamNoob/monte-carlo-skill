# MCMC / Bayesian Monte Carlo Reference

## When to use MCMC vs. plain MC

| Situation | Method |
|-----------|--------|
| Prior × Likelihood → posterior, low-dimensional (<10 params), independent/weakly correlated | Metropolis-Hastings (below) |
| Correlated or higher-dimensional posterior — M-H mixes slowly | HMC/NUTS, see `hmc-nuts.md` (measured 19-113x effective-sample-size gain over M-H on a 0.9-correlation target) |
| Real Bayesian modelling with observed data | Use PyMC or Stan |
| Learning / pedagogy | Implement M-H from scratch |
| Sampling from known distribution (normalising constant unknown) | MCMC |
| Forward uncertainty propagation | Plain MC (not MCMC) |

## Metropolis-Hastings from Scratch

```python
import numpy as np

def metropolis_hastings(log_prob, initial, n_samples, proposal_std=0.5, seed=42):
    """
    General M-H sampler.
    
    log_prob: callable(theta) → log unnormalised posterior
    initial: starting point (array or scalar)
    proposal_std: std of Gaussian random walk proposal
    
    Returns: samples array, acceptance rate
    """
    rng = np.random.default_rng(seed)
    theta = np.array(initial, dtype=float)
    samples = np.zeros((n_samples, theta.size))
    accepted = 0
    
    log_p_current = log_prob(theta)
    
    for i in range(n_samples):
        # Propose
        proposal = theta + rng.normal(0, proposal_std, theta.shape)
        log_p_proposal = log_prob(proposal)
        
        # Accept / reject
        log_alpha = log_p_proposal - log_p_current
        if np.log(rng.uniform()) < log_alpha:
            theta = proposal
            log_p_current = log_p_proposal
            accepted += 1
        
        samples[i] = theta
    
    return samples, accepted / n_samples

# Example: Gaussian posterior
def log_posterior_example(theta):
    mu, log_sigma = theta
    sigma = np.exp(log_sigma)
    data = np.array([2.1, 1.9, 2.3, 2.0, 2.2])
    log_likelihood = -0.5 * np.sum((data - mu)**2) / sigma**2 - len(data) * log_sigma
    log_prior_mu = -0.5 * mu**2  # N(0, 1) prior on mu
    # N(0, 1) prior on log_sigma (i.e. a lognormal prior on sigma itself).
    # Without this term sigma has an implicit flat prior in log-space —
    # improper (doesn't integrate to 1), which happens to still produce a
    # valid posterior here because the likelihood is informative enough,
    # but is easy to get away with by accident on a smaller dataset. State
    # a proper prior explicitly rather than relying on that.
    log_prior_log_sigma = -0.5 * log_sigma**2
    return log_likelihood + log_prior_mu + log_prior_log_sigma


def run_chains(log_prob, init_fn, n_chains, n_samples, proposal_std=0.5, seed=42):
    """
    Run n_chains independent Metropolis-Hastings chains from dispersed
    starting points (required for R-hat / ESS diagnostics to mean anything
    — a single chain can't detect its own non-convergence).

    init_fn: callable(rng) -> initial theta, should sample from an
        over-dispersed distribution relative to the expected posterior.
    Returns: array of shape (n_chains, n_samples, n_params), acceptance rates
    """
    rng = np.random.default_rng(seed)
    all_samples = []
    acc_rates = []
    for c in range(n_chains):
        init = init_fn(rng)
        chain_seed = rng.integers(0, 2**31 - 1)
        samples, acc = metropolis_hastings(log_prob, init, n_samples, proposal_std, chain_seed)
        all_samples.append(samples)
        acc_rates.append(acc)
    return np.stack(all_samples), np.array(acc_rates)

samples, acc_rates = run_chains(
    log_posterior_example,
    init_fn=lambda rng: rng.normal(0, 3, size=2),  # over-dispersed vs. N(0,1) prior
    n_chains=4, n_samples=10_000,
)
print(f"Acceptance rates: {acc_rates}")  # aim for 23-50% each
# Burn-in: discard first 20% from each chain
samples_post_burnin = samples[:, 2000:, :]

# Feed into arviz for R-hat / ESS (see Diagnostics below):
# import arviz as az
# idata = az.convert_to_inference_data(samples_post_burnin)
```

## PyMC (recommended for real problems)

```python
import pymc as pm
import numpy as np

data = np.array([2.1, 1.9, 2.3, 2.0, 2.2])

with pm.Model() as model:
    # Priors
    mu = pm.Normal("mu", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)
    
    # Likelihood
    obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=data)
    
    # Sample
    trace = pm.sample(2000, tune=1000, chains=4, return_inferencedata=True)

pm.plot_posterior(trace, var_names=["mu", "sigma"])
```

## MCMC Diagnostics

Always check these before trusting MCMC results:

```python
import arviz as az

# R-hat: should be < 1.01 for all parameters
print(az.summary(trace, round_to=3))

# Effective sample size: should be > 400 per chain
# Trace plots: should look like "fuzzy caterpillars"
az.plot_trace(trace)

# Autocorrelation: should decay quickly
az.plot_autocorr(trace)
```

## Acceptance Rate Tuning

- Target 23% for high-dimensional problems (>5 dims)
- Target 44% for 1D problems
- If too high (>70%): increase `proposal_std` — proposals too small, chain doesn't explore
- If too low (<10%): decrease `proposal_std` — proposals too large, almost always rejected

## Common MCMC Pitfalls

- **Not discarding burn-in**: discard first 20-50% of samples
- **Single chain**: always run 4+ chains; if they don't mix, your sampler is stuck
- **Thinning by default**: only thin if you're constrained by memory; ESS matters more than raw sample count
- **Ignoring R-hat**: if R-hat > 1.05 for any parameter, the chains haven't converged
- **Sampling in constrained space**: use log-transform for σ>0, logit-transform for p∈(0,1)
