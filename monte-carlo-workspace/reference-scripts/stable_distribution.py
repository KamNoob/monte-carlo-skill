#!/usr/bin/env python3
"""
Alpha-stable (Levy stable) distribution fitting and Monte Carlo sampling.

The alpha-stable family generalises the normal distribution with two extra
parameters, alpha (tail index, 0 < alpha <= 2) and beta (skewness,
-1 <= beta <= 1). alpha=2 is exactly Gaussian; alpha<2 gives power-law tails
(P(|X|>x) ~ x^-alpha) heavier than any lognormal/normal model, which is the
point of using it: equity/FX daily log-returns empirically fit alpha in
~1.5-1.9, and a Gaussian or lognormal model (as used by gbm_stock_sim.py)
systematically understates tail risk for that data. See McCulloch (1986) for
the quantile-based estimator and Nolan (2020, "Univariate Stable
Distributions") for the general theory.

Two hazards specific to this family, both handled below:
  1. For alpha < 2, the population variance is infinite (and undefined for
     alpha <= 1). Sample mean/std computed on a stable-distributed series is
     not a consistent estimator of anything — never report "volatility" from
     one the way gbm_stock_sim.py does. Use quantile-based risk measures
     (VaR/CVaR from the empirical/simulated quantiles) instead.
  2. scipy's `fit()` uses McCulloch's quantile estimator for a starting
     point, then refines by MLE. The docs warn MLE does not always converge
     when using the FFT pdf method and alpha <= 1 — this script forces the
     robust 'piecewise' pdf method (the default) rather than opting into FFT.

Usage:
    python3 stable_distribution.py --demo
    python3 stable_distribution.py --returns-csv returns.csv --confidence 0.99
"""

import argparse
import contextlib
import csv
import json
import math
import sys
import time
from typing import Optional

import numpy as np
from scipy import stats


# scipy.stats.levy_stable is a process-wide singleton: setting
# `.parameterization`/`.pdf_default_method` mutates global state, not a
# local view. Save/restore around every use so this script can't leave the
# global in a different state than it found it (or silently disagree with
# other code in the same process that also touches this distribution).
@contextlib.contextmanager
def _s0_piecewise():
    dist = stats.levy_stable
    prev_param = dist.parameterization
    prev_pdf_method = dist.pdf_default_method
    dist.parameterization = "S0"
    dist.pdf_default_method = "piecewise"  # robust method; MLE isn't guaranteed to converge under FFT for alpha<=1
    try:
        yield dist
    finally:
        dist.parameterization = prev_param
        dist.pdf_default_method = prev_pdf_method


# scipy's MLE refinement can converge to a degenerate boundary (alpha
# pinned near 2, beta pinned near +-1) instead of a genuine optimum —
# observed directly: n=100 synthetic draws from alpha=1.7 fit to exactly
# alpha=2.0, beta=-1.0. That's a syntactically valid 4-tuple that silently
# turns a heavy-tail model into a near-Gaussian/max-skew one. Reject fits
# that land on/near either boundary rather than returning them unflagged.
_BOUNDARY_EPS = 1e-3


def fit_stable(returns: np.ndarray) -> dict:
    """
    Fit alpha, beta, loc, scale to a 1-D array of returns via scipy's
    McCulloch-quantile + MLE-refinement estimator (S0 parameterization).

    Requires at least 250 observations. McCulloch's quantile estimator reads
    off the 5th/25th/50th/75th/95th percentiles as a starting point for MLE;
    empirically even n=500 recovered beta with ~40% error against a known
    true value, so this is a floor to avoid the worst noise, not a
    guarantee of a tight estimate — always sanity-check the fitted alpha
    against prior expectations for the asset class (see
    references/stable-distributions.md's validation section).
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1:
        raise ValueError("returns must be a 1-D array")
    if not np.all(np.isfinite(returns)):
        raise ValueError("returns must not contain NaN/inf")
    if len(returns) < 250:
        raise ValueError("need at least 250 observations for a stable fit to be meaningfully reliable")

    # Measured ~70s for 500 points on reference hardware (MLE refinement
    # repeatedly evaluates the numerically-integrated piecewise pdf) —
    # warn rather than let this look like a hang for larger inputs.
    print(
        f"fitting alpha-stable to {len(returns)} points via McCulloch quantile "
        "+ MLE — this can take tens of seconds to minutes, scipy has no "
        "faster exact method for this distribution",
        file=sys.stderr,
    )
    t0 = time.monotonic()
    with _s0_piecewise() as dist:
        alpha, beta, loc, scale = dist.fit(returns)
    print(f"fit finished in {time.monotonic() - t0:.1f}s", file=sys.stderr)

    if alpha >= 2 - _BOUNDARY_EPS or alpha <= _BOUNDARY_EPS or abs(beta) >= 1 - _BOUNDARY_EPS:
        raise ValueError(
            f"fit collapsed to a boundary value (alpha={alpha:.4f}, beta={beta:.4f}) "
            "instead of a genuine optimum — this is a known scipy MLE failure mode, "
            "not a real result; try more observations or a different random seed's "
            "worth of data before trusting a fit this close to (0,2]x[-1,1]'s edge"
        )

    return {"alpha": float(alpha), "beta": float(beta), "loc": float(loc), "scale": float(scale)}


def simulate_stable_returns(
    alpha: float,
    beta: float,
    loc: float,
    scale: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample (n_paths, n_steps) iid stable-distributed returns via scipy's
    rvs(), which implements the Chambers-Mallows-Stuck transform internally.

    Returns are NOT cumulatively summed into a price path here: for alpha<2
    the stable family is still additive (sums of iid stable(alpha,beta) are
    stable(alpha,beta) at a rescaled scale), so summing log-returns to get a
    terminal log-price is valid, but the caller must decide the summing
    horizon explicitly rather than this function silently picking one.
    """
    if seed is not None and rng is not None:
        raise ValueError("seed and rng are mutually exclusive")
    if not (0 < alpha <= 2):
        raise ValueError("alpha must be in (0, 2]")
    if not (-1 <= beta <= 1):
        raise ValueError("beta must be in [-1, 1]")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive number")
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be positive integers")

    if rng is None:
        rng = np.random.default_rng(seed)

    with _s0_piecewise() as dist:
        samples = dist.rvs(alpha, beta, loc=loc, scale=scale, size=(n_paths, n_steps), random_state=rng)
    return samples


def stable_vs_normal_var_cvar(
    returns: np.ndarray,
    horizon_steps: int,
    confidence: float = 0.99,
    n_paths: int = 100_000,
    seed: int = 42,
) -> dict:
    """
    Fit both a stable and a normal distribution to `returns`, Monte Carlo
    the horizon-step cumulative log-return under each, and report VaR/CVaR
    from both — the gap is the tail risk a Gaussian/lognormal model misses.

    VaR/CVaR here are quantile-based (percentile of simulated outcomes), not
    moment-based, so they stay well-defined even though the stable fit's
    population variance is infinite for alpha<2.
    """
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0, 1)")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")

    stable_params = fit_stable(returns)
    rng = np.random.default_rng(seed)

    stable_samples = simulate_stable_returns(
        stable_params["alpha"],
        stable_params["beta"],
        stable_params["loc"],
        stable_params["scale"],
        n_steps=horizon_steps,
        n_paths=n_paths,
        rng=rng,
    )
    stable_cum_return = stable_samples.sum(axis=1)

    mu_norm, sigma_norm = float(np.mean(returns)), float(np.std(returns, ddof=1))
    normal_cum_return = rng.normal(mu_norm * horizon_steps, sigma_norm * math.sqrt(horizon_steps), n_paths)

    def var_cvar(x: np.ndarray) -> tuple:
        var = -np.percentile(x, (1 - confidence) * 100)
        tail = x[x <= -var]
        cvar = -tail.mean() if len(tail) > 0 else float("nan")
        return float(var), float(cvar), len(tail)

    stable_var, stable_cvar, stable_tail_n = var_cvar(stable_cum_return)
    normal_var, normal_cvar, normal_tail_n = var_cvar(normal_cum_return)

    result = {
        "stable_params": stable_params,
        "normal_params": {"mu": mu_norm, "sigma": sigma_norm},
        "confidence": confidence,
        "horizon_steps": horizon_steps,
        "stable_var": stable_var,
        "stable_cvar": stable_cvar,
        "normal_var": normal_var,
        "normal_cvar": normal_cvar,
        "var_ratio_stable_over_normal": stable_var / normal_var if normal_var != 0 else float("nan"),
    }
    for label, n in (("stable", stable_tail_n), ("normal", normal_tail_n)):
        if n < 30:
            result.setdefault("warnings", []).append(
                f"{label} tail has only {n} samples at this confidence — CVaR is noisy, increase n_paths"
            )
    return result


def _load_returns_csv(path: str) -> np.ndarray:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    values = []
    for row in rows:
        if not row:
            continue
        try:
            values.append(float(row[0]))
        except ValueError:
            continue  # header row or blank
    if not values:
        raise ValueError(f"no numeric values found in {path}")
    return np.array(values, dtype=float)


def _demo_returns(seed: int = 7) -> np.ndarray:
    """
    Synthetic alpha=1.7 stable log-returns, standing in for real equity data.
    250 points: the minimum fit_stable() will accept. scipy's
    levy_stable.fit() MLE step is slow and its runtime doesn't scale
    predictably with n (measured 55s-127s across n=200-500 on this machine)
    — see the timing note printed before fit() runs.
    """
    with _s0_piecewise() as dist:
        return dist.rvs(1.7, 0.0, loc=0.0, scale=0.01, size=250, random_state=np.random.default_rng(seed))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit an alpha-stable distribution and compare tail risk to Gaussian.")
    parser.add_argument("--returns-csv", type=str, default=None, help="Path to a single-column CSV of returns (e.g. daily log-returns)")
    parser.add_argument("--demo", action="store_true", help="Use synthetic alpha=1.7 stable returns instead of a CSV")
    parser.add_argument("--horizon-steps", type=int, default=5, help="Number of steps to cumulate for the VaR/CVaR horizon (default 5)")
    parser.add_argument("--confidence", type=float, default=0.99, help="VaR/CVaR confidence level (default 0.99)")
    parser.add_argument("--paths", type=int, default=100_000, help="Monte Carlo paths (default 100000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    args = parser.parse_args()

    if bool(args.returns_csv) == bool(args.demo):
        print("error: pass exactly one of --returns-csv or --demo", file=sys.stderr)
        raise SystemExit(1)

    try:
        if args.demo:
            # Synthetic data is fully under our control, so if scipy's MLE
            # happens to collapse to a boundary fit (a real, observed
            # failure mode — see fit_stable's docstring) just draw a fresh
            # demo sample rather than surfacing that as a demo failure.
            # Real user data (--returns-csv) never gets this treatment —
            # a boundary collapse there is a genuine result to report, not
            # something to paper over by resampling.
            last_exc = None
            result = None
            for attempt in range(5):
                try:
                    returns = _demo_returns(seed=7 + attempt)
                    result = stable_vs_normal_var_cvar(
                        returns,
                        horizon_steps=args.horizon_steps,
                        confidence=args.confidence,
                        n_paths=args.paths,
                        seed=args.seed,
                    )
                    break
                except ValueError as exc:
                    if "boundary value" not in str(exc):
                        raise
                    last_exc = exc
                    print(f"demo attempt {attempt+1}/5 hit a boundary fit, retrying with a new sample", file=sys.stderr)
            if result is None:
                raise last_exc
        else:
            returns = _load_returns_csv(args.returns_csv)
            result = stable_vs_normal_var_cvar(
                returns,
                horizon_steps=args.horizon_steps,
                confidence=args.confidence,
                n_paths=args.paths,
                seed=args.seed,
            )
    except (ValueError, OverflowError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
