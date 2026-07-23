#!/usr/bin/env python3
"""
Variance-Gamma (VG) process: simulation, method-of-moments fitting, and
tail-risk comparison against Gaussian.

VG is a subclass of the generalized hyperbolic (GH) family — the "semi-
heavy-tailed" alternative to lognormal/GBM returns, sitting between GBM
(no skew/excess kurtosis by construction) and the alpha-stable family
(references/stable-distributions.md; heavier, power-law tails, possibly
infinite variance). VG has finite moments of all orders but nonzero
skewness and excess kurtosis controllable independently — useful for
returns that are clearly non-Gaussian (fat shoulders, asymmetry) but
where infinite-variance tails would be too extreme a claim. It underlies
the Madan-Carr-Chang (1998) option pricing model and traces to Madan &
Seneta (1990), "The Variance Gamma (V.G.) Model for Share Market Returns".

WHY NOT scipy.stats.genhyperbolic: scipy implements the general GH family,
of which VG is a boundary/limiting case (delta -> 0 in the (lambda,alpha,
beta,delta) parameterization) rather than a generic interior point.
scipy's own docs warn: "For distributions that are a special case [of
genhyperbolic] such as Student's t, it is not recommended to rely on the
implementation of genhyperbolic... the methods of the specific
distributions should be used" — the same caveat applies to VG. This
script implements VG directly via its own construction (a Gamma-
subordinated Brownian motion), which needs no Bessel functions for
sampling and has closed-form moments, avoiding scipy's boundary-case
numerical fragility entirely.

Construction: X_t = mu*t + theta*G_t + sigma*W(G_t), where G_t is a Gamma
process (G_t ~ Gamma(shape=t/nu, scale=nu), E[G_t]=t, Var[G_t]=t*nu) and
W is an independent standard Brownian motion. theta controls skew
(theta<0 => left skew, matching the typical asymmetry of equity returns),
nu controls excess kurtosis (nu=0 degenerates to Gaussian), sigma is the
base volatility scale.

Moment formulas below were DERIVED from the cumulant generating function
K(u) = -(t/nu)*log(1 - nu*theta*u - 0.5*nu*sigma^2*u^2) (a standard
Levy-process result: MGF of a normal variance-mean mixture with a Gamma
subordinator, via the tower property and the Gamma MGF) and verified
against 3-million-path simulation before being trusted here (mean/var/
skew/kurtosis all matched to within Monte Carlo noise) — not taken from
an unverified literature formula.

Usage:
    python3 variance_gamma.py --demo
"""

import argparse
import json
import math
import sys
from typing import Optional

import numpy as np
from scipy import optimize, stats


def vg_cumulants(sigma: float, nu: float, theta: float, t: float = 1.0) -> dict:
    """
    Closed-form mean/variance/skewness/excess-kurtosis of a VG(sigma, nu,
    theta) increment over horizon t, derived from the process's cumulant
    generating function (see module docstring) and verified against
    simulation.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if nu <= 0:
        raise ValueError("nu must be positive (nu=0 is the degenerate Gaussian limit)")
    if t <= 0:
        raise ValueError("t must be positive")
    var_total = sigma**2 + nu * theta**2
    mean = t * theta
    var = t * var_total
    skew = nu * theta * (3 * sigma**2 + 2 * nu * theta**2) / (math.sqrt(t) * var_total**1.5)
    excess_kurt = 3 * nu * (sigma**4 + 4 * nu * theta**2 * sigma**2 + 2 * nu**2 * theta**4) / (t * var_total**2)
    return {"mean": mean, "var": var, "skew": skew, "excess_kurtosis": excess_kurt}


def simulate_vg(sigma: float, nu: float, theta: float, mu: float, t: float, n_paths: int, seed: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Sample n_paths iid VG(sigma, nu, theta) increments over horizon t,
    plus an added drift mu*t. Single-shot sampling (not a stepped path)
    since VG increments over disjoint intervals are independent — a path
    of daily increments can be built by repeated calls with t=1, or a
    single call with t=n_days directly gives the same distribution as the
    sum of n_days iid unit increments (Levy process additivity), whichever
    is needed; this function returns only the terminal increment.
    """
    if seed is not None and rng is not None:
        raise ValueError("seed and rng are mutually exclusive")
    if sigma <= 0 or nu <= 0 or t <= 0 or n_paths < 1:
        raise ValueError("sigma, nu, t must be positive and n_paths >= 1")
    if rng is None:
        rng = np.random.default_rng(seed)
    G = rng.gamma(t / nu, nu, n_paths)
    return mu * t + theta * G + sigma * np.sqrt(G) * rng.standard_normal(n_paths)


def fit_vg_moments(returns: np.ndarray) -> dict:
    """
    Method-of-moments fit: match sample (variance, skewness, excess
    kurtosis) to vg_cumulants' formulas via a 2-D nonlinear solve for
    (nu, theta) — sigma is eliminated via the variance identity
    sigma^2 = var/t - nu*theta^2, and mu is recovered from the mean
    afterward. t=1 (fit is per-observation-unit, matching whatever
    period the input returns are sampled at).

    Deliberately avoids MLE via the Bessel-function VG density (which
    exists in closed form but is numerically delicate near the same
    boundary cases scipy's genhyperbolic warns about) — this is a
    simpler, more robust estimator appropriate for a reference
    implementation, at the cost of statistical efficiency vs. full MLE.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1 or len(returns) < 30:
        raise ValueError("returns must be a 1-D array with at least 30 observations")
    if not np.all(np.isfinite(returns)):
        raise ValueError("returns must not contain NaN/inf")

    m, v = returns.mean(), returns.var(ddof=1)
    # bias=False for consistency with the ddof=1 (unbiased) variance above —
    # mixing an unbiased variance with scipy's default biased (population,
    # n-normalized) skew/kurtosis estimators would feed inconsistent moment
    # targets into the same solve, most consequential exactly in the
    # small-sample / near-Gaussian regime the nu<0.05 warning below exists for.
    skew = stats.skew(returns, bias=False)
    kurt = stats.kurtosis(returns, bias=False)  # excess (Fisher) kurtosis

    def equations(params):
        nu, theta = params
        if nu <= 1e-8 or v <= 0:
            return [1e6, 1e6]
        sigma2 = v - nu * theta**2  # implied sigma^2 from the variance identity var = sigma^2 + nu*theta^2
        if sigma2 <= 1e-12:
            return [1e6, 1e6]
        skew_pred = nu * theta * (3 * v - nu * theta**2) / v**1.5
        kurt_pred = 3 * nu * (sigma2**2 + 4 * nu * theta**2 * sigma2 + 2 * nu**2 * theta**4) / v**2
        return [skew_pred - skew, kurt_pred - kurt]

    nu0 = max(abs(kurt), 0.3) / 3
    theta0 = skew * math.sqrt(v) / (3 * nu0)
    sol = optimize.root(equations, x0=[nu0, theta0], method="hybr")
    if not sol.success:
        raise ValueError(
            f"method-of-moments solve did not converge (sample skew={skew:.4f}, "
            f"excess kurtosis={kurt:.4f}) — this (skew, kurtosis) pair may not be "
            "reachable by any valid VG(sigma>0, nu>0) combination; try more data "
            "or a different distribution family"
        )
    nu, theta = sol.x
    sigma2 = v - nu * theta**2
    if nu <= 0 or sigma2 <= 0:
        raise ValueError(
            f"method-of-moments solve returned an invalid fit (nu={nu:.4f}, "
            f"implied sigma^2={sigma2:.4f}) — not a valid VG parameterization"
        )
    sigma = math.sqrt(sigma2)
    mu = m - theta  # t=1

    result = {"sigma": float(sigma), "nu": float(nu), "theta": float(theta), "mu": float(mu)}

    # This estimator is genuinely ill-conditioned near the Gaussian
    # boundary: even a true Gaussian population has nonzero SAMPLE
    # kurtosis from finite-sample noise, which forces nu away from 0, and
    # then matching even a small realized skew forces theta to compensate
    # (theta enters the skew formula divided by nu, so small nu needs
    # large theta to hit the same skew target). Verified empirically: a
    # 5000-point exactly-Gaussian sample (skew=0.055, excess kurt=0.089,
    # both near-zero but not exactly zero as expected) fit to nu=0.029,
    # theta=0.64 — a mathematically exact root (residual ~1e-14, confirmed
    # against 25+ different solver starting points, all converging to the
    # same point), not a solver artifact, but a practically misleading fit
    # for what is actually near-Gaussian data. Flag rather than silently
    # return this.
    if nu < 0.05:
        result.setdefault("warnings", []).append(
            f"fitted nu ({nu:.4f}) is small, meaning the sample excess kurtosis was "
            "close to zero (near-Gaussian) — this is a known instability zone of "
            "moment-matching: any nonzero sample skew (even from finite-sample "
            f"noise alone) then forces a disproportionately large theta ({theta:.4f}) "
            "to compensate, since theta enters the skew formula divided by nu. "
            "Check whether a simple Gaussian model already fits this data adequately "
            "before trusting this VG fit."
        )
    # The solve is also ill-conditioned at the OTHER extreme: as true nu
    # grows, VG's higher moments (which the (skew,kurt) equations depend
    # on) get increasingly noisy in finite samples — verified empirically
    # by fitting synthetic data across a range of true nu: recovery is
    # good up to nu~1, visibly biased by nu~3 (sigma undershoots the true
    # value), and the solve fails outright by nu~15 (sample skew/kurtosis
    # too noisy to be a reachable VG moment pair at all, correctly raising
    # ValueError above rather than returning a bad fit silently — but the
    # degrading region in between, nu~3-8, succeeds without any flag).
    if nu > 3:
        result.setdefault("warnings", []).append(
            f"fitted nu ({nu:.4f}) is large — VG's higher sample moments become "
            "noisy as nu grows, so this fit is less reliable than the residual "
            "alone suggests; consider more data or checking recovery on a "
            "synthetic sample with similar (skew, kurtosis) before trusting it"
        )
    return result


def vg_vs_normal_var_cvar(returns: np.ndarray, horizon_steps: int, confidence: float = 0.99, n_paths: int = 100_000, seed: int = 42) -> dict:
    """
    Fit VG and Normal to `returns`, Monte Carlo the horizon cumulative
    return under each (VG increments are additive over disjoint periods,
    so horizon_steps unit increments sum correctly), report VaR/CVaR from
    both — same pattern as stable_distribution.py's stable_vs_normal_var_cvar.
    """
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0, 1)")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")

    vg_params = fit_vg_moments(returns)
    rng = np.random.default_rng(seed)

    vg_cum = simulate_vg(vg_params["sigma"], vg_params["nu"], vg_params["theta"], vg_params["mu"], t=horizon_steps, n_paths=n_paths, rng=rng)

    mu_norm, sigma_norm = float(np.mean(returns)), float(np.std(returns, ddof=1))
    normal_cum = rng.normal(mu_norm * horizon_steps, sigma_norm * math.sqrt(horizon_steps), n_paths)

    def var_cvar(x):
        var = -np.percentile(x, (1 - confidence) * 100)
        tail = x[x <= -var]
        cvar = -tail.mean() if len(tail) > 0 else float("nan")
        return float(var), float(cvar), len(tail)

    vg_var, vg_cvar, vg_tail_n = var_cvar(vg_cum)
    normal_var, normal_cvar, normal_tail_n = var_cvar(normal_cum)

    result = {
        "vg_params": vg_params,
        "normal_params": {"mu": mu_norm, "sigma": sigma_norm},
        "confidence": confidence,
        "horizon_steps": horizon_steps,
        "vg_var": vg_var, "vg_cvar": vg_cvar,
        "normal_var": normal_var, "normal_cvar": normal_cvar,
        "var_ratio_vg_over_normal": vg_var / normal_var if normal_var != 0 else float("nan"),
    }
    for label, n in (("vg", vg_tail_n), ("normal", normal_tail_n)):
        if n < 30:
            result.setdefault("warnings", []).append(f"{label} tail has only {n} samples at this confidence — CVaR is noisy, increase n_paths")
    return result


def _demo_returns(seed: int = 7) -> np.ndarray:
    """Synthetic returns from a known VG(sigma=0.012, nu=0.3, theta=-0.004) — mild negative skew, fat shoulders."""
    return simulate_vg(sigma=0.012, nu=0.3, theta=-0.004, mu=0.0, t=1.0, n_paths=2000, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a Variance-Gamma process and compare tail risk to Gaussian.")
    parser.add_argument("--returns-csv", type=str, default=None, help="Path to a single-column CSV of returns")
    parser.add_argument("--demo", action="store_true", help="Use synthetic VG returns instead of a CSV")
    parser.add_argument("--horizon-steps", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if bool(args.returns_csv) == bool(args.demo):
        print("error: pass exactly one of --returns-csv or --demo", file=sys.stderr)
        raise SystemExit(1)

    try:
        if args.demo:
            returns = _demo_returns()
        else:
            with open(args.returns_csv, newline="") as f:
                import csv
                values = []
                for row in csv.reader(f):
                    if not row:
                        continue
                    try:
                        values.append(float(row[0]))
                    except ValueError:
                        continue
                if not values:
                    raise ValueError(f"no numeric values found in {args.returns_csv}")
                returns = np.array(values, dtype=float)
        result = vg_vs_normal_var_cvar(returns, horizon_steps=args.horizon_steps, confidence=args.confidence, n_paths=args.paths, seed=args.seed)
    except (ValueError, OverflowError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
