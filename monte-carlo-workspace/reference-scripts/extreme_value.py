#!/usr/bin/env python3
"""
Extreme Value Theory: GEV (block maxima) and GPD (peaks-over-threshold),
with POT-derived VaR/CVaR at confidence levels far beyond what raw
percentiles or a GBM/stable/VG fit can reliably estimate.

Where this fits: stable_distribution.py and variance_gamma.py fit a full
return distribution and read off tail percentiles from it — accurate
near the body but the fit is doing double duty (matching the whole shape
AND the tail). EVT instead fits ONLY the tail, using the Pickands-Balkema-
de Haan theorem: for a threshold u high enough, exceedances X-u | X>u are
asymptotically Generalized Pareto distributed *regardless of the parent
distribution* — the same GPD applies whether the underlying returns are
Gaussian, stable, or VG. This is the right tool specifically for "how bad
could the 1-in-1000 day be" questions where the confidence level is far
out past what the bulk of the data can inform, and the wrong tool for
everyday VaR (95-99%) where a full-distribution fit is more efficient and
doesn't waste data by discarding everything below the threshold.

Two classical approaches:
  GEV (Generalized Extreme Value): fit to BLOCK MAXIMA (e.g. the worst
    day in each month). Answers "what's the distribution of the single
    worst observation in a period of this length."
  GPD (Generalized Pareto): fit to EXCEEDANCES over a threshold (peaks-
    over-threshold, POT). Answers "given we're already past threshold u,
    how much further could it go" — and is what this script uses for
    VaR/CVaR, since POT uses far more of the tail data than block maxima
    (one point per block vs. every exceedance) for the same amount of
    history.

CRITICAL SCIPY GOTCHA, verified empirically (not just from docs): scipy's
two extreme-value distributions use OPPOSITE shape-parameter sign
conventions relative to each other.
  - scipy.stats.genpareto's `c` parameter matches the standard EVT shape
    xi directly (fit c=0.248 recovered a true xi=0.25 with no sign flip).
  - scipy.stats.genextreme's `c` is the NEGATIVE of the standard xi —
    scipy's own docs flag this explicitly ("several sources... use the
    opposite convention"), and it was confirmed here by checking which
    sign of c gives an unbounded-above support (the correct behavior for
    xi>0/Frechet-type heavy tails): c=-0.25 was unbounded above, c=+0.25
    was bounded above — i.e. scipy_genextreme_c == -xi_standard.
  This script converts scipy's genextreme output to standard xi
  internally (see fit_gev) specifically so a caller comparing a GEV shape
  to a GPD shape (which the Pickands-Balkema-de Haan theorem says should
  be the SAME xi for the same underlying tail) doesn't get bitten by a
  silent sign flip between the two.

Usage:
    python3 extreme_value.py --demo
"""

import argparse
import json
import math
import sys
from typing import Optional

import numpy as np
from scipy import stats


def fit_gev(block_maxima: np.ndarray) -> dict:
    """
    Fit GEV to a sample of block maxima (e.g. one value per month/year).
    Returns standard-convention xi (shape), NOT scipy's raw `c` — see
    module docstring for the verified sign-flip gotcha this corrects for.
    xi>0: heavy (Frechet) tail, unbounded. xi<0: bounded (Weibull-type)
    tail. xi=0: Gumbel (light, exponential-like) tail.
    """
    block_maxima = np.asarray(block_maxima, dtype=float)
    if block_maxima.ndim != 1 or len(block_maxima) < 20:
        raise ValueError("block_maxima must be a 1-D array with at least 20 blocks")
    if not np.all(np.isfinite(block_maxima)):
        raise ValueError("block_maxima must not contain NaN/inf")
    c, loc, scale = stats.genextreme.fit(block_maxima)
    xi = -c  # convert scipy's flipped convention to standard EVT xi
    return {"xi": float(xi), "loc": float(loc), "scale": float(scale), "n_blocks": len(block_maxima)}


def fit_gpd(data: np.ndarray, threshold: float, n_bootstrap: int = 200, seed: Optional[int] = None) -> dict:
    """
    Fit GPD to exceedances of `data` over `threshold` (peaks-over-
    threshold). Returns standard-convention xi (scipy's genpareto `c`
    already matches xi directly, no flip needed — verified empirically,
    see module docstring). loc is fixed at 0 (fitting the exceedances
    Y=X-u, not X itself) since the POT theorem's asymptotic result is
    stated for the exceedance distribution starting at 0.

    Requires at least 100 exceedances — NOT a stability guarantee, just a
    floor against outright degenerate fits. Verified empirically (300-
    trial simulation, true xi=0.25): at 25 exceedances, fitted xi had
    std~0.28 — roughly as large as the parameter itself, with the fitted
    sign flipping (bounded vs. unbounded tail!) routinely; at 500
    exceedances the estimate is still visibly biased low with std~0.05.
    xi is a genuinely high-variance parameter to estimate from tail data
    — always check `xi_ci` (a bootstrap confidence interval, included
    below) rather than trusting the point estimate alone, especially for
    Nu in the low hundreds rather than high hundreds/thousands.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("data must be a 1-D array")
    if not np.all(np.isfinite(data)):
        raise ValueError("data must not contain NaN/inf")
    exceedances = data[data > threshold] - threshold
    n_exceed = len(exceedances)
    if n_exceed < 100:
        raise ValueError(
            f"only {n_exceed} exceedances above threshold {threshold} — need at least 100 "
            "to avoid a degenerate fit (and even 100 has real estimation uncertainty, see "
            "xi_ci in the returned dict); lower the threshold or supply more data"
        )
    xi, loc, sigma = stats.genpareto.fit(exceedances, floc=0)

    rng = np.random.default_rng(seed)
    boot_xi, boot_sigma = np.empty(n_bootstrap), np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(exceedances, size=n_exceed, replace=True)
        boot_xi[i], _, boot_sigma[i] = stats.genpareto.fit(resample, floc=0)
    xi_ci = (float(np.percentile(boot_xi, 2.5)), float(np.percentile(boot_xi, 97.5)))
    sigma_ci = (float(np.percentile(boot_sigma, 2.5)), float(np.percentile(boot_sigma, 97.5)))

    return {
        "xi": float(xi), "sigma": float(sigma), "threshold": float(threshold),
        "n_total": len(data), "n_exceedances": n_exceed,
        "exceedance_rate": n_exceed / len(data),
        "xi_ci_95": xi_ci, "sigma_ci_95": sigma_ci,
        "_bootstrap_xi_sigma": list(zip(boot_xi.tolist(), boot_sigma.tolist())),
    }


def _var_cvar_formula(xi: float, sigma: float, u: float, n: int, Nu: int, confidence: float) -> tuple:
    """Pure VaR/CVaR computation from GPD parameters — factored out so gpd_var_cvar can reuse it per-bootstrap-replicate for a confidence interval, not just the point estimate."""
    tail_prob = (1 - confidence) * n / Nu
    if tail_prob >= 1:
        raise ValueError(
            f"confidence={confidence} is not far enough into the tail for this threshold "
            f"(implied tail_prob={tail_prob:.4f} >= 1, i.e. VaR would land below the "
            "fitting threshold) — use a lower threshold or a higher confidence"
        )
    if abs(xi) < 1e-8:
        var = u - sigma * math.log(tail_prob)
    else:
        var = u + (sigma / xi) * (tail_prob ** (-xi) - 1)
    cvar = float("inf") if xi >= 1 else var / (1 - xi) + (sigma - xi * u) / (1 - xi)
    return float(var), float(cvar)


def gpd_var_cvar(gpd_fit: dict, confidence: float) -> dict:
    """
    VaR/CVaR at `confidence` from a GPD peaks-over-threshold fit, derived
    from the POT tail-probability formula P(X>u+y) = (Nu/n)*(1+xi*y/sigma)^(-1/xi)
    and (for CVaR) the GPD's threshold-stability property (a GPD's
    exceedances above any higher threshold are themselves GPD with the
    same xi and a shifted sigma). Both formulas were verified end-to-end
    against two different known distributions' TRUE VaR/CVaR (computed
    directly, not from another fit): Student-t(df=4) (VaR within 0.57%,
    CVaR within 1.38% at 99.9% confidence) and Pareto Type I (VaR within
    0.24%, CVaR within 0.14%).

    If `gpd_fit` came from `fit_gpd` (and so includes bootstrap
    replicates), this also reports a 95% CI on VaR/CVaR by evaluating the
    same formula at each bootstrap (xi, sigma) pair — xi is a genuinely
    high-variance parameter to estimate (see fit_gpd's docstring), and
    that uncertainty propagates directly into VaR/CVaR; a lone point
    estimate here would understate how much is actually known.

    Requires confidence high enough that VaR lands above the fitting
    threshold — this function only extrapolates outward from the
    threshold, per the POT theorem's own domain of validity.
    """
    xi, sigma, u = gpd_fit["xi"], gpd_fit["sigma"], gpd_fit["threshold"]
    n, Nu = gpd_fit["n_total"], gpd_fit["n_exceedances"]
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0, 1)")
    var, cvar = _var_cvar_formula(xi, sigma, u, n, Nu, confidence)
    result = {"confidence": confidence, "var": var, "cvar": cvar, "tail_prob_at_threshold": Nu / n}

    boot = gpd_fit.get("_bootstrap_xi_sigma")
    if boot:
        boot_var, boot_cvar = [], []
        for boot_xi, boot_sigma in boot:
            try:
                v, c = _var_cvar_formula(boot_xi, boot_sigma, u, n, Nu, confidence)
                boot_var.append(v)
                boot_cvar.append(c)
            except ValueError:
                continue  # this replicate's tail_prob went out of range; skip rather than distort the CI
        if boot_var:
            result["var_ci_95"] = (float(np.percentile(boot_var, 2.5)), float(np.percentile(boot_var, 97.5)))
            result["cvar_ci_95"] = (float(np.percentile(boot_cvar, 2.5)), float(np.percentile(boot_cvar, 97.5)))
    return result


def choose_threshold_percentile(data: np.ndarray, percentile: float = 95.0) -> float:
    """
    Simple default threshold: a fixed percentile of the data. This is a
    starting point, not a substitute for a mean-residual-life plot (the
    standard EVT diagnostic — plot mean exceedance vs. threshold, look for
    where it becomes roughly linear) which requires visual judgment and
    isn't automated here. The bias-variance tradeoff: too low a threshold
    violates the POT theorem's asymptotic assumption (bias from including
    non-tail data); too high leaves too few exceedances (high variance in
    the fit). 90-95th percentile is a common starting point for daily
    financial return data; always sanity-check against a range of nearby
    thresholds (see the demo's stability check).
    """
    data = np.asarray(data, dtype=float)
    if not (0 < percentile < 100):
        raise ValueError("percentile must be in (0, 100)")
    return float(np.percentile(data, percentile))


def _demo() -> dict:
    """
    Ground-truth check: simulate from a KNOWN Student-t (df=4, heavy-
    tailed) distribution, fit GPD via POT, and compare the GPD-derived
    VaR/CVaR at a deep tail confidence (99.9%) against the true values
    computed directly from the known distribution — the same validation
    done during development, reproduced here so the demo is a genuine
    correctness check, not just "does it run."
    """
    rng = np.random.default_rng(3)
    df = 4
    n = 500_000
    data = stats.t.rvs(df, size=n, random_state=rng)

    threshold = choose_threshold_percentile(data, 95.0)
    # n_bootstrap=50 (vs fit_gpd's default 200) keeps the demo's runtime
    # reasonable at n_exceedances=25000 — real usage on smaller datasets
    # can afford the full default.
    gpd_fit = fit_gpd(data, threshold, n_bootstrap=50, seed=1)
    confidence = 0.999
    result = gpd_var_cvar(gpd_fit, confidence)

    true_var = float(stats.t.ppf(confidence, df))
    from scipy import integrate
    true_cvar = float(integrate.quad(lambda x: x * stats.t.pdf(x, df), true_var, np.inf)[0] / (1 - confidence))

    # threshold-stability sanity check: refit at a nearby threshold, xi should be similar
    threshold2 = choose_threshold_percentile(data, 97.0)
    gpd_fit2 = fit_gpd(data, threshold2, n_bootstrap=50, seed=2)

    gpd_fit_summary = {k: v for k, v in gpd_fit.items() if not k.startswith("_")}

    return {
        "true_distribution": f"Student-t(df={df})",
        "n_observations": n,
        "gpd_fit": gpd_fit_summary,
        "gpd_var_cvar_at_99.9pct": result,
        "true_var_99.9pct": true_var,
        "true_cvar_99.9pct": true_cvar,
        "var_relative_error_pct": abs(result["var"] - true_var) / true_var * 100,
        "cvar_relative_error_pct": abs(result["cvar"] - true_cvar) / true_cvar * 100,
        "threshold_stability_check": {
            "xi_at_95th_percentile_threshold": gpd_fit["xi"],
            "xi_at_97th_percentile_threshold": gpd_fit2["xi"],
            "note": "xi should be reasonably stable across nearby thresholds if the POT approximation is valid here",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit GPD via peaks-over-threshold and validate VaR/CVaR against a known distribution.")
    parser.add_argument("--demo", action="store_true", required=True)
    parser.parse_args()
    try:
        result = _demo()
    except (ValueError, OverflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
