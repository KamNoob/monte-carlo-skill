#!/usr/bin/env python3
"""
Copula-based dependence sampling: Gaussian, Student-t, and Clayton.

correlated_gbm.py models cross-asset dependence via a correlation matrix
applied to *normal* shocks — that's implicitly a Gaussian copula on
lognormal (GBM) marginals. This script generalizes the *dependence
structure itself*, independent of marginals: a copula is a joint
distribution on [0,1]^n with uniform marginals, capturing only "how
variables move together," which you then feed through any marginal
distribution's inverse CDF (ppf) to build the actual joint model —
including marginals correlated_gbm.py can't use, like the alpha-stable
returns from stable_distribution.py.

Three copula families, in increasing order of tail-dependence realism:

  Gaussian copula: no tail dependence — even at high correlation, the
    probability of a *joint extreme* event goes to zero faster than the
    marginals would suggest. This is the copula implicit in
    correlated_gbm.py and in a plain multivariate-normal VaR model
    (financial.md's portfolio_var_cvar). Famously blamed (rightly or
    with some oversimplification) for underestimating joint mortgage
    defaults in the 2008 financial crisis specifically because of this
    zero-tail-dependence property.

  Student-t copula: same correlation-matrix structure as Gaussian, plus
    a degrees-of-freedom parameter controlling *symmetric* tail
    dependence — lower df means assets are more likely to crash (or
    spike) together than a Gaussian copula would predict, at the same
    linear correlation.

  Clayton copula (Archimedean, single parameter alpha): *asymmetric*
    tail dependence — strong lower-tail dependence (assets crash
    together) but weak upper-tail dependence (assets don't rally
    together as strongly). This asymmetry is a well-documented empirical
    feature of real equity returns and is exactly the pattern a Gaussian
    or t-copula (both symmetric) cannot represent at all.

Gumbel and Frank (the other two common Archimedean copulas) are not
implemented here: Gumbel's frailty variable is itself a positive
alpha-stable random variable (see stable_distribution.py — same
Chambers-Mallows-Stuck sampling problem, plus scipy has no built-in
one-sided/positive-stable sampler), and Frank's frailty needs sampling
from a Logarithmic distribution with no scipy built-in either. Both are
real gaps, noted rather than silently worked around.

Usage:
    python3 copula_sampling.py --demo
"""

import argparse
import json
import math
import sys
from typing import Callable, Optional

import numpy as np
from scipy import stats


def gaussian_copula_sample(corr_matrix: np.ndarray, n_samples: int, seed: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Sample from a Gaussian copula: correlated standard-normal draws via
    Cholesky (same construction as correlated_gbm.py), transformed to
    [0,1] marginals via the standard normal CDF. Returns (n_samples, n)
    array of uniform-marginal, Gaussian-dependence samples.
    """
    if seed is not None and rng is not None:
        raise ValueError("seed and rng are mutually exclusive")
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    n = corr_matrix.shape[0]
    if rng is None:
        rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr_matrix)
    z = rng.standard_normal((n_samples, n)) @ L.T
    return stats.norm.cdf(z)


def t_copula_sample(corr_matrix: np.ndarray, df: float, n_samples: int, seed: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Sample from a Student-t copula: correlated normal draws divided by
    sqrt(chi2(df)/df) — the standard normal-variance-mixture construction
    of a multivariate t — then transformed via the univariate t CDF at
    the same df. Lower df => fatter, more symmetric tail dependence than
    Gaussian at the same correlation matrix.
    """
    if seed is not None and rng is not None:
        raise ValueError("seed and rng are mutually exclusive")
    if df <= 0:
        raise ValueError("df must be positive")
    corr_matrix = np.asarray(corr_matrix, dtype=float)
    n = corr_matrix.shape[0]
    if rng is None:
        rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr_matrix)
    z = rng.standard_normal((n_samples, n)) @ L.T
    chi2 = rng.chisquare(df, n_samples)
    t_samples = z / np.sqrt(chi2 / df)[:, None]  # (n_samples, n), each row a correlated multivariate-t draw
    return stats.t.cdf(t_samples, df=df)


def clayton_copula_sample(alpha: float, n_assets: int, n_samples: int, seed: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Sample from an exchangeable n-dimensional Clayton copula via the
    Marshall-Olkin frailty method: draw a shared frailty V ~ Gamma(1/alpha, 1)
    per sample, then n independent Exponential(1) draws E_i, and set
    U_i = (1 + E_i/V)^(-1/alpha). All pairs share the same alpha
    (exchangeable structure) — a real limitation vs. a full pairwise
    correlation matrix, noted rather than hidden.

    alpha > 0 required; alpha -> 0 approaches independence, larger alpha
    means stronger (lower-tail) dependence. Kendall's tau = alpha/(alpha+2).
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if seed is not None and rng is not None:
        raise ValueError("seed and rng are mutually exclusive")
    if n_assets < 2:
        raise ValueError("n_assets must be at least 2")
    if rng is None:
        rng = np.random.default_rng(seed)
    v = rng.gamma(1.0 / alpha, 1.0, n_samples)
    e = rng.exponential(1.0, (n_samples, n_assets))
    return (1.0 + e / v[:, None]) ** (-1.0 / alpha)


def gaussian_corr_from_kendalls_tau(tau: np.ndarray) -> np.ndarray:
    """Elliptical-copula (Gaussian/t) relationship: rho = sin(pi/2 * tau). Same formula for both families since it only depends on the correlation matrix, not df."""
    return np.sin(np.pi / 2 * np.asarray(tau, dtype=float))


def clayton_alpha_from_kendalls_tau(tau: float) -> float:
    """Clayton: tau = alpha/(alpha+2) => alpha = 2*tau/(1-tau). Requires 0 < tau < 1 (Clayton has no negative-dependence range in this parameterization)."""
    if not (0 < tau < 1):
        raise ValueError("Clayton copula only supports 0 < tau < 1 (positive dependence)")
    return 2 * tau / (1 - tau)


def apply_marginals(uniforms: np.ndarray, ppfs: list) -> np.ndarray:
    """
    Transform a copula's uniform-marginal samples into the target joint
    distribution by applying each column's own inverse-CDF (ppf). This is
    the actual point of using a copula: uniforms[:, i] carries only the
    dependence structure; ppfs[i] carries that asset's own marginal shape
    (which need not match any other asset's — mixing a stable marginal
    with a lognormal one is perfectly valid here).

    ppfs: list of callables, one per column, each mapping an array of
    values in (0,1) to that column's marginal quantiles.
    """
    if uniforms.shape[1] != len(ppfs):
        raise ValueError(f"uniforms has {uniforms.shape[1]} columns but {len(ppfs)} ppf functions given")
    return np.column_stack([ppf(uniforms[:, i]) for i, ppf in enumerate(ppfs)])


def _demo() -> dict:
    rng = np.random.default_rng(42)
    n_samples = 200_000
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    tau_target = 0.6
    alpha = clayton_alpha_from_kendalls_tau(tau_target)
    corr_matched = np.array([[1.0, gaussian_corr_from_kendalls_tau(tau_target)], [gaussian_corr_from_kendalls_tau(tau_target), 1.0]])

    gauss_u = gaussian_copula_sample(corr_matched, n_samples, rng=rng)
    t_u = t_copula_sample(corr_matched, df=4, n_samples=n_samples, rng=rng)
    clay_u = clayton_copula_sample(alpha, 2, n_samples, rng=rng)

    def lower_tail_dependence(u: np.ndarray, q: float = 0.05) -> float:
        """Empirical P(U2 < q | U1 < q) — should agree across copulas only if they have matched tail behavior, which Gaussian/t/Clayton at the same Kendall's tau do NOT."""
        below_u1 = u[:, 0] < q
        if below_u1.sum() == 0:
            return float("nan")
        return float((u[below_u1, 1] < q).mean())

    return {
        "kendalls_tau_target": tau_target,
        "gaussian_corr_used": float(corr_matched[0, 1]),
        "clayton_alpha_used": alpha,
        "empirical_kendalls_tau": {
            "gaussian": float(stats.kendalltau(gauss_u[:5000, 0], gauss_u[:5000, 1])[0]),
            "t": float(stats.kendalltau(t_u[:5000, 0], t_u[:5000, 1])[0]),
            "clayton": float(stats.kendalltau(clay_u[:5000, 0], clay_u[:5000, 1])[0]),
        },
        "empirical_lower_tail_dependence_at_q0.05": {
            "gaussian": lower_tail_dependence(gauss_u),
            "t_df4": lower_tail_dependence(t_u),
            "clayton": lower_tail_dependence(clay_u),
        },
        "note": (
            "All three copulas were tuned to the SAME Kendall's tau (0.6), yet their "
            "joint-crash probabilities differ substantially — that gap is the entire "
            "point of choosing a copula family deliberately rather than defaulting to "
            "Gaussian dependence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Copula dependence sampling demo.")
    parser.add_argument("--demo", action="store_true", required=True, help="Run the built-in Gaussian/t/Clayton tail-dependence comparison")
    parser.parse_args()
    try:
        result = _demo()
    except (ValueError, OverflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
