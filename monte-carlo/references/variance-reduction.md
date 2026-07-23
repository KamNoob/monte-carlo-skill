# Variance Reduction Techniques

Use these when plain sampling gives estimates that converge too slowly — typically when:
- Estimating rare-event probabilities (< 1%)
- The function being estimated has high variance
- You need high precision with a fixed computational budget

## 1. Antithetic Variates

**Idea**: for every random sample U, also use 1-U (or -Z for normals). The two outputs are negatively correlated, so their average has lower variance.

```python
import numpy as np

def mc_antithetic(f, N, rng=None):
    """
    Estimate E[f(U)] where U ~ Uniform(0,1) using antithetic variates.
    Uses N/2 independent samples, doubled via antithesis.
    """
    rng = rng or np.random.default_rng(42)
    U = rng.uniform(0, 1, N // 2)
    results = (f(U) + f(1 - U)) / 2
    return results.mean(), results.std() / np.sqrt(N // 2)

# For normal samples: use Z and -Z
def mc_antithetic_normal(f, N, rng=None):
    rng = rng or np.random.default_rng(42)
    Z = rng.standard_normal(N // 2)
    results = (f(Z) + f(-Z)) / 2
    return results.mean(), results.std() / np.sqrt(N // 2)
```

Best for: smooth, monotone functions. Gain: 2–10× variance reduction typical.

## 2. Control Variates

**Idea**: if you know E[g(X)] analytically, use g(X) as a control to reduce variance in E[f(X)].

```python
def mc_control_variate(f, g, E_g, X_samples):
    """
    f: target function
    g: control function with known expectation E_g
    X_samples: pre-generated samples
    """
    f_vals = f(X_samples)
    g_vals = g(X_samples)
    
    # Optimal coefficient
    c_star = -np.cov(f_vals, g_vals)[0, 1] / np.var(g_vals)
    
    adjusted = f_vals + c_star * (g_vals - E_g)
    return adjusted.mean(), adjusted.std() / np.sqrt(len(X_samples))

# Example: estimate E[e^U] using control g(U) = U with E[U] = 0.5
import numpy as np
rng = np.random.default_rng(42)
U = rng.uniform(0, 1, 100_000)
est, se = mc_control_variate(np.exp, lambda u: u, 0.5, U)
# True answer: e - 1 ≈ 1.7183
```

Best for: when a correlated function with known expectation exists. Gain: can be 10–100× for well-chosen controls.

## 3. Importance Sampling

**Idea**: sample from a proposal distribution q(x) that concentrates samples where the integrand f(x)p(x) is large. Correct via importance weights.

Two variants — pick based on whether `p` is a normalized density:

```python
def importance_sampling(f, p_log_prob, q_sample, q_log_prob, N, seed=42):
    """
    Standard IS: estimate E_p[f(X)] = ∫ f(x) p(x) dx by sampling from q.
    Use this ONLY when p_log_prob is a properly normalized log-density
    (integrates to 1). Weights are NOT renormalized to sum to 1 — that
    would silently turn this into the (biased, for finite N) self-normalized
    estimator below.

    p_log_prob: log p(x) function (must be normalized)
    q_sample: function(N, seed) → N samples from q
    q_log_prob: log q(x) function
    """
    X = q_sample(N, seed)
    log_weights = p_log_prob(X) - q_log_prob(X)
    weights = np.exp(log_weights)

    estimate = np.mean(weights * f(X))
    std_err = np.std(weights * f(X), ddof=1) / np.sqrt(N)
    ess = np.sum(weights)**2 / np.sum(weights**2)  # effective sample size

    return estimate, std_err, ess


def self_normalized_importance_sampling(f, p_log_prob, q_sample, q_log_prob, N, seed=42):
    """
    Self-normalized IS: use when p_log_prob is only known up to a constant
    (unnormalized target — the common case in Bayesian posteriors). Biased
    for finite N (bias -> 0 as N -> inf), but doesn't require knowing p's
    normalizing constant. This is what the previous version of this file
    computed unconditionally, which is correct here but WRONG if you meant
    the standard estimator above.
    """
    X = q_sample(N, seed)
    log_weights = p_log_prob(X) - q_log_prob(X)
    log_weights -= log_weights.max()  # for numerical stability only
    weights = np.exp(log_weights)
    weights /= weights.sum()

    estimate = np.sum(weights * f(X))
    ess = 1 / np.sum(weights**2)  # effective sample size (weights sum to 1 here)

    return estimate, ess

# Effective sample size (ESS) should be > 10% of N; if lower, q is a bad fit.
# Support check: q(x) must be > 0 everywhere f(x)p(x) is nonzero, or the
# estimator is biased with no way to detect it from the samples alone.
```

Best for: rare events, tails of distributions. Gain: can be orders of magnitude for rare-event problems.

**Warning**: bad proposal distributions can make IS *worse* than plain MC. The proposal should have heavier tails than the target.

## 4. Stratified Sampling

**Idea**: divide the sample space into strata and sample a fixed number from each. Eliminates clustering.

```python
def stratified_uniform(N, n_strata=None, rng=None):
    """Generate N stratified uniform samples in [0,1]."""
    rng = rng or np.random.default_rng(42)
    n_strata = n_strata or N
    k, remainder = divmod(N, n_strata)
    # Distribute the remainder across the first `remainder` strata instead
    # of silently dropping those samples (the original `N // n_strata`
    # truncation loses up to n_strata-1 samples with no warning).
    counts = np.full(n_strata, k)
    counts[:remainder] += 1

    strata_starts = np.repeat(np.arange(n_strata) / n_strata, counts)
    within_strata = rng.uniform(0, 1 / n_strata, N)
    samples = strata_starts + within_strata
    rng.shuffle(samples)  # break the strata-order structure before use
    return samples
```

Best for: low-dimensional integrals where you can partition the domain. Gain: O(1/N) convergence instead of O(1/√N) for smooth integrands.

For more than a few dimensions, prefer **Latin Hypercube Sampling** (`scipy.stats.qmc.LatinHypercube`) or **quasi-Monte Carlo** (below) over stratifying each axis independently — the number of strata needed grows exponentially with dimension otherwise.

## 5. Quasi-Monte Carlo (low-discrepancy sequences)

**Idea**: replace pseudo-random samples with a deterministic low-discrepancy sequence (Sobol, Halton) that fills the space more evenly than random points do. Convergence improves from O(1/√N) toward O(1/N) for smooth, low-to-moderate dimensional integrands.

```python
from scipy.stats import qmc

def qmc_sobol(f, d, N, seed=42):
    """
    Estimate E[f(U)] for U ~ Uniform([0,1]^d) via a scrambled Sobol sequence.
    N should be a power of 2 for Sobol's balance guarantees to hold —
    sampler.random(N) still works otherwise but loses that property.
    """
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    U = sampler.random(N)  # shape (N, d)
    vals = f(U)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(N)  # SE formula is only
    # a rough guide for QMC (samples aren't iid) — for a real error bound,
    # run several independently-scrambled sequences and use their spread.
```

Best for: smooth integrands in low-to-moderate dimensions (roughly <20). Scrambling (`scramble=True`) is required to get valid randomized error estimates — an unscrambled sequence is fully deterministic and gives no way to quantify error from a single run. Push inputs through `scipy.stats.norm.ppf(U)` etc. to convert the `[0,1]^d` samples into other distributions.

## Choosing the Right Technique

| Situation | Recommended |
|-----------|------------|
| General, easy to implement | Antithetic variates |
| Have analytical knowledge of related function | Control variates |
| Rare event (probability < 0.01) | Importance sampling |
| Low-dimensional smooth integral | Stratified sampling |
| No idea, just want less variance | Antithetic variates first |

Always report the **variance reduction factor**: `var_plain / var_reduced`. A 10× reduction means you need 10× fewer samples for the same precision.
