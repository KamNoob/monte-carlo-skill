# European Call Option Pricing — Monte Carlo with Black-Scholes Validation

## Problem Classification

This is a **Financial Monte Carlo** problem (Type 3 from the skill). We are pricing a European call option, which has a known analytical solution (Black-Scholes), making it an ideal validation case.

---

## Parameters

| Parameter | Value |
|-----------|-------|
| Current stock price (S0) | £100.00 |
| Strike price (K) | £105.00 |
| Risk-free rate (r) | 5% per annum |
| Volatility (σ) | 20% per annum |
| Time to expiry (T) | 0.5 years (6 months) |
| Option type | European call |

---

## Simulation Design

**Random input:** Terminal stock price S_T under the risk-neutral measure.

**Model:** Geometric Brownian Motion (GBM) — the standard assumption in Black-Scholes theory. Under the risk-neutral measure, the drift is the risk-free rate r (not the real-world drift μ). This is the standard approach for pricing derivatives.

**Terminal price formula:**

```
S_T = S0 * exp((r - 0.5 * σ²) * T  +  σ * √T * Z)
```

where Z ~ N(0,1).

This is the exact GBM terminal distribution — no discretisation error, since European options only depend on the terminal price (not the path).

**Estimator:** `payoff = max(S_T - K, 0)`, discounted at `exp(-r * T)`.

**Sample count:** 500,000 — chosen to achieve relative standard error below 0.3%.

---

## Implementation

```python
import numpy as np
import math

# Parameters
S0 = 100.0      # current stock price (£)
K = 105.0       # strike price (£)
r = 0.05        # risk-free rate (annual)
sigma = 0.20    # volatility (annual)
T = 0.5         # time to expiry in years (6 months)
N = 500_000     # number of simulations

rng = np.random.default_rng(seed=42)

# --- Monte Carlo: simulate terminal stock prices ---
Z = rng.standard_normal(N)
S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

# --- Compute payoffs and discount ---
payoffs = np.maximum(S_T - K, 0)
discount = np.exp(-r * T)

mc_price = discount * payoffs.mean()
std_err   = discount * payoffs.std() / np.sqrt(N)

ci_lo = mc_price - 1.96 * std_err
ci_hi = mc_price + 1.96 * std_err


# --- Black-Scholes analytical formula (validation) ---
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
d2 = d1 - sigma * math.sqrt(T)
bs_price = S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
```

---

## Results

### Monte Carlo Price

| Metric | Value |
|--------|-------|
| MC price | **£4.5841** |
| Standard error | £0.011587 |
| 95% confidence interval | **[£4.5614, £4.6068]** |
| Relative standard error | 0.253% |
| Simulations | 500,000 |

### Black-Scholes Analytical Price

| Metric | Value |
|--------|-------|
| BS price | **£4.5817** |
| d1 | -0.097511 |
| d2 | -0.238933 |
| N(d1) | 0.461160 |
| N(d2) | 0.405579 |

### Validation

| Check | Result |
|-------|--------|
| Absolute error (MC vs BS) | £0.0024 |
| Relative error | 0.053% |
| BS price inside 95% CI | **Yes** |

The Black-Scholes price of £4.5817 sits comfortably inside the MC confidence interval [£4.5614, £4.6068]. The MC estimate is within 0.05% of the analytical truth — well within acceptable precision.

---

## Black-Scholes Derivation (for reference)

The Black-Scholes formula for a European call is:

```
C = S0 * N(d1) - K * exp(-r*T) * N(d2)

d1 = [ln(S0/K) + (r + σ²/2)*T] / (σ*√T)
d2 = d1 - σ*√T
```

With our parameters:
- `d1 = [ln(100/105) + (0.05 + 0.02)*0.5] / (0.20 * 0.7071)`
- `d1 = [-0.04879 + 0.035] / 0.14142 = -0.09751`
- `d2 = -0.09751 - 0.14142 = -0.23893`
- `N(-0.09751) = 0.4612`, `N(-0.23893) = 0.4056`
- `C = 100 * 0.4612 - 105 * exp(-0.025) * 0.4056`
- `C = 46.12 - 105 * 0.97531 * 0.4056 = 46.12 - 41.54 = £4.58`

---

## Convergence Analysis

Confirming the MC estimate converges to the BS price as N increases:

| Simulations (N) | MC Price | Std Error | Error vs BS |
|-----------------|----------|-----------|-------------|
| 1,000 | £4.2197 | £0.2486 | £0.362 |
| 10,000 | £4.5350 | £0.0821 | £0.047 |
| 100,000 | £4.5766 | £0.0260 | £0.005 |
| 500,000 | £4.5841 | £0.0116 | £0.002 |

Convergence follows the expected O(1/√N) rate. At N=500,000 the error is negligible for practical purposes.

---

## Key Assumptions

1. **Risk-neutral measure**: drift set to r=5% (not real-world stock drift). This is correct for derivative pricing under no-arbitrage.
2. **Constant volatility**: σ=20% is constant. In reality, implied volatility varies by strike/expiry (volatility smile). For more accuracy, use a stochastic vol model (e.g., Heston).
3. **No dividends**: the stock pays no dividends. If there were a continuous dividend yield q, replace r with r-q in the GBM drift.
4. **GBM model**: stock follows log-normal diffusion — the same assumption as Black-Scholes, so perfect agreement is expected.
5. **European exercise**: payoff depends only on terminal price; no early exercise.

---

## Summary

The Monte Carlo price is **£4.5841** with a 95% confidence interval of **[£4.5614, £4.6068]**, using 500,000 simulations. The Black-Scholes analytical price is **£4.5817**, sitting inside the CI. The absolute error is £0.0024 (0.05%) — effectively noise at this sample size.

The option is slightly out-of-the-money (S0=£100 < K=£105), which is why it trades below £5. The 6-month horizon and 20% volatility give it meaningful time value.

To improve precision further: increase N (1M would halve the standard error), or apply **antithetic variates** (simulate Z and -Z pairs) which typically halves variance at no extra cost.
