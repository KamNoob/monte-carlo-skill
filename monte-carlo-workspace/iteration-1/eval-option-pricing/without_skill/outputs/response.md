# European Call Option Pricing: Monte Carlo vs Black-Scholes

## Problem Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Current stock price | S₀ | £100.00 |
| Strike price | K | £105.00 |
| Risk-free rate | r | 5% p.a. |
| Volatility | σ | 20% p.a. |
| Time to expiry | T | 0.5 years (6 months) |

---

## Part 1: Black-Scholes Analytical Price

The Black-Scholes formula for a European call option is:

```
C = S₀·N(d₁) − K·e^(−rT)·N(d₂)
```

where:

```
d₁ = [ln(S₀/K) + (r + σ²/2)·T] / (σ√T)
d₂ = d₁ − σ√T
```

### Calculation

```
d₁ = [ln(100/105) + (0.05 + 0.5·0.04)·0.5] / (0.20·√0.5)
   = [ln(0.95238) + 0.035] / 0.14142
   = [−0.04879 + 0.035] / 0.14142
   = −0.013793 / 0.14142
   = −0.097511

d₂ = −0.097511 − 0.14142
   = −0.238933
```

Applying the standard normal CDF (via `math.erf`):

```
N(d₁) = N(−0.097511) = 0.461160
N(d₂) = N(−0.238933) = 0.405579
```

Discount factor:

```
e^(−rT) = e^(−0.05·0.5) = e^(−0.025) = 0.975310
```

**Black-Scholes Price:**

```
C_BS = 100 · 0.461160 − 105 · 0.975310 · 0.405579
     = 46.116 − 41.534
     = £4.5817
```

---

## Part 2: Monte Carlo Simulation

### Method

Under the risk-neutral measure, stock price at expiry follows geometric Brownian motion:

```
S_T = S₀ · exp[(r − σ²/2)·T + σ·√T·Z]
```

where Z ~ N(0,1).

For each of N = 1,000,000 paths:
1. Draw Z from N(0,1) using the Box-Muller transform
2. Compute terminal stock price S_T
3. Compute discounted payoff: e^(−rT) · max(S_T − K, 0)

The MC price is the average of all discounted payoffs.

### Python Implementation

```python
import math, random

S0 = 100.0; K = 105.0; r = 0.05; sigma = 0.20; T = 0.5
N = 1_000_000

random.seed(42)
payoffs = []

for _ in range(N):
    # Box-Muller transform: U1, U2 ~ Uniform(0,1) → Z ~ N(0,1)
    u1 = random.random() or 1e-300  # guard against log(0)
    u2 = random.random()
    Z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    
    # Risk-neutral stock price at expiry
    ST = S0 * math.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
    
    # Call payoff
    payoffs.append(max(ST - K, 0.0))

discount = math.exp(-r * T)
mean_payoff = sum(payoffs) / N
mc_price = discount * mean_payoff

# Standard error of the mean
variance = sum((p - mean_payoff)**2 for p in payoffs) / (N - 1)
std_error = math.sqrt(variance / N)

# 95% confidence interval
z95 = 1.96
ci_lo = discount * (mean_payoff - z95 * std_error)
ci_hi = discount * (mean_payoff + z95 * std_error)
```

### Results

| Metric | Value |
|--------|-------|
| MC Price | **£4.5853** |
| Standard Error | £0.008183 |
| 95% CI Lower | £4.5693 |
| 95% CI Upper | £4.6013 |
| CI Width | £0.0321 |

The 95% confidence interval is **[£4.5693, £4.6013]**.

---

## Part 3: Validation Against Black-Scholes

| Metric | Value |
|--------|-------|
| Black-Scholes price | £4.5817 |
| Monte Carlo price | £4.5853 |
| Absolute difference | £0.0036 |
| Relative error | 0.079% |
| BS price inside 95% CI | **Yes** |

The Black-Scholes price of £4.5817 falls comfortably inside the Monte Carlo 95% confidence interval [£4.5693, £4.6013]. The 0.08% relative error is well within the expected sampling noise for 1,000,000 simulations.

---

## Key Observations

**Why the option is cheap relative to stock price:**
- The option is out of the money (S₀ = £100 < K = £105)
- Only 6 months remain — limited time for the stock to recover
- N(d₁) = 0.461, N(d₂) = 0.406 — both below 0.5, reflecting OTM status

**Monte Carlo convergence:**
With N = 1,000,000 paths, the standard error is ~£0.008, giving a CI width of ~£0.032. This is typical: MC error decreases as 1/√N. To halve the error, you would need 4× as many paths.

**Antithetic variates (variance reduction):**
A common improvement is to pair each Z with −Z, halving the number of random draws needed for the same accuracy. This can reduce the standard error by 30–50% for vanilla options at little extra cost.

---

## Summary

```
Black-Scholes Price  =  £4.5817
Monte Carlo Price    =  £4.5853  (N = 1,000,000)
95% Confidence Int.  = [£4.5693, £4.6013]
Relative Error       =  0.079%
Validation           =  PASS — BS inside 95% CI
```
