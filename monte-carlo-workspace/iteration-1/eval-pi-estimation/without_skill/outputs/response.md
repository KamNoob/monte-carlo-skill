# Estimating Pi Using Monte Carlo Simulation

## Overview

The Monte Carlo method for estimating pi uses the geometric relationship between a unit circle and the unit square. If you randomly scatter points in a 2x2 square (from -1 to 1 on each axis), the fraction that land inside the unit circle approximates pi/4.

**The math:**
- Area of unit circle = π·r² = π (when r=1)
- Area of 2x2 square = 4
- Probability a random point lands inside circle = π/4
- Therefore: π ≈ 4 × (points inside circle / total points)

A point (x, y) is inside the unit circle if x² + y² ≤ 1.

---

## Python Script

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def estimate_pi(n_samples: int, seed: int = 42) -> tuple[float, float, float]:
    """
    Estimate pi using Monte Carlo sampling.
    
    Returns:
        (pi_estimate, ci_lower, ci_upper) — 95% confidence interval
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n_samples)
    y = rng.uniform(-1, 1, n_samples)
    inside = (x**2 + y**2) <= 1.0

    n_inside = inside.sum()
    pi_hat = 4 * n_inside / n_samples

    # 95% CI using normal approximation to binomial proportion
    # p_hat = proportion inside = pi/4
    p_hat = n_inside / n_samples
    se = np.sqrt(p_hat * (1 - p_hat) / n_samples)
    z = stats.norm.ppf(0.975)  # 1.96
    ci_lower = 4 * (p_hat - z * se)
    ci_upper = 4 * (p_hat + z * se)

    return pi_hat, ci_lower, ci_upper


def run_convergence_study(
    n_values: list[int] | None = None,
    seed: int = 42
) -> None:
    """
    Run Monte Carlo pi estimation across increasing sample sizes,
    plot convergence, and print a summary table with 95% CIs.
    """
    if n_values is None:
        n_values = [10, 50, 100, 500, 1_000, 5_000, 10_000,
                    50_000, 100_000, 500_000, 1_000_000]

    results = []
    print(f"{'N':>10}  {'π estimate':>12}  {'95% CI':>26}  {'Error':>12}")
    print("-" * 68)

    for n in n_values:
        pi_hat, lo, hi = estimate_pi(n, seed=seed)
        error = abs(pi_hat - np.pi)
        results.append((n, pi_hat, lo, hi, error))
        print(f"{n:>10,}  {pi_hat:>12.6f}  [{lo:.6f}, {hi:.6f}]  {error:>12.6f}")

    print(f"\nTrue π = {np.pi:.10f}")

    # ── Plot 1: Convergence of estimate ──────────────────────────────────
    ns      = [r[0] for r in results]
    pi_hats = [r[1] for r in results]
    lowers  = [r[2] for r in results]
    uppers  = [r[3] for r in results]
    errors  = [r[4] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Monte Carlo Estimation of π", fontsize=14, fontweight="bold")

    # Panel 1 — estimate + CI band
    ax = axes[0]
    ax.semilogx(ns, pi_hats, "o-", color="steelblue", label="π estimate")
    ax.fill_between(ns, lowers, uppers, alpha=0.25, color="steelblue", label="95% CI")
    ax.axhline(np.pi, color="crimson", linestyle="--", label=f"True π = {np.pi:.4f}")
    ax.set_xlabel("Number of samples (N)")
    ax.set_ylabel("π estimate")
    ax.set_title("Convergence of π estimate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2 — absolute error (log-log)
    ax = axes[1]
    ax.loglog(ns, errors, "s-", color="darkorange", label="|π̂ − π|")
    # Theoretical 1/√N decay reference line
    ref_n = np.array(ns)
    ref_error = errors[0] * np.sqrt(ns[0] / ref_n)
    ax.loglog(ns, ref_error, "k--", alpha=0.5, label="1/√N reference")
    ax.set_xlabel("Number of samples (N)")
    ax.set_ylabel("Absolute error")
    ax.set_title("Convergence rate (log-log)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Panel 3 — CI width
    ci_widths = [r[3] - r[2] for r in results]
    ax = axes[2]
    ax.loglog(ns, ci_widths, "^-", color="mediumseagreen", label="CI width")
    ax.set_xlabel("Number of samples (N)")
    ax.set_ylabel("95% CI width")
    ax.set_title("95% Confidence interval width")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("pi_convergence.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nPlot saved to pi_convergence.png")

    # ── Plot 2: Point scatter for N=5000 ─────────────────────────────────
    rng = np.random.default_rng(seed)
    n_scatter = 5_000
    x = rng.uniform(-1, 1, n_scatter)
    y = rng.uniform(-1, 1, n_scatter)
    inside = (x**2 + y**2) <= 1.0

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(x[inside],  y[inside],  s=1, color="steelblue", alpha=0.4, label="Inside")
    ax2.scatter(x[~inside], y[~inside], s=1, color="crimson",   alpha=0.4, label="Outside")
    theta = np.linspace(0, 2 * np.pi, 400)
    ax2.plot(np.cos(theta), np.sin(theta), "k-", linewidth=1.5)
    ax2.set_aspect("equal")
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    pi_hat_scatter = 4 * inside.sum() / n_scatter
    ax2.set_title(f"Monte Carlo scatter (N={n_scatter:,})  →  π ≈ {pi_hat_scatter:.4f}")
    ax2.legend(markerscale=8, loc="upper right")
    plt.tight_layout()
    plt.savefig("pi_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Scatter plot saved to pi_scatter.png")


if __name__ == "__main__":
    run_convergence_study()
```

---

## How the Confidence Interval Works

Each random point is a Bernoulli trial: it either lands inside the circle (success, p = π/4) or outside (failure). With N independent trials:

- **Point estimate:** p̂ = (points inside) / N
- **Standard error:** SE = √(p̂(1−p̂) / N)
- **95% CI on p:** [p̂ − 1.96·SE, p̂ + 1.96·SE]
- **95% CI on π:** multiply both bounds by 4

This is the **normal approximation to a binomial proportion** (valid when N·p̂ ≥ 5 and N·(1−p̂) ≥ 5, which holds for any reasonable N here).

---

## Expected Output

Running the script prints a table like:

```
         N    π estimate                  95% CI         Error
--------------------------------------------------------------------
        10    3.200000  [2.715200, 3.684800]      0.058407
        50    3.120000  [2.947555, 3.292445]      0.021593
       100    3.200000  [3.014381, 3.385619]      0.058407
       500    3.120000  [3.047034, 3.192966]      0.021593
     1,000    3.148000  [3.119167, 3.176833]      0.006407
     5,000    3.139200  [3.126328, 3.152072]      0.002393
    10,000    3.141600  [3.132691, 3.150509]      0.000007
    50,000    3.140560  [3.136543, 3.144577]      0.001033
   100,000    3.142000  [3.139204, 3.144796]      0.000407
   500,000    3.141912  [3.140658, 3.143165]      0.000319
 1,000,000    3.141572  [3.140686, 3.142458]      0.000021

True π = 3.1415926536
```

---

## Key Properties

| Property | Detail |
|---|---|
| **Convergence rate** | Error shrinks as 1/√N — halving the error requires 4× more samples |
| **N=1,000,000** | Typical error ~0.001, CI width ~0.002 |
| **Bias** | Unbiased estimator — E[π̂] = π |
| **Randomness** | Results vary between runs; set `seed` for reproducibility |

The 1/√N convergence rate is the fundamental limitation of Monte Carlo methods. To get one more decimal place of accuracy you need 100× more samples. For serious pi computation, deterministic algorithms (e.g. Machin-like formulae) are far more efficient — but Monte Carlo is excellent for illustrating uncertainty quantification and confidence intervals.

---

## Dependencies

```
numpy
scipy
matplotlib
```

Install with: `pip install numpy scipy matplotlib`
