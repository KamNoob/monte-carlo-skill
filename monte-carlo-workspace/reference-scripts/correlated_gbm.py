#!/usr/bin/env python3
"""
Correlated multi-asset GBM Monte Carlo simulator.

gbm_stock_sim.py simulates one asset at a time, each with its own
independent standard-normal shock. That's wrong for a portfolio: real
assets move together (NVDA/AMD share sector risk, gold/silver share
precious-metals risk), and treating them as independent understates
portfolio-level tail risk — a joint 2-sigma drawdown across correlated
assets is far more likely than independent sampling implies. This script
extends the same GBM model to a portfolio via a correlation matrix and
Cholesky decomposition, so the ticker-by-ticker code path (this doesn't
replace gbm_stock_sim.py — running one asset at a time is still correct
and simpler when no cross-asset question is being asked) gets a
correlated counterpart for the (fairly common) case where it is.

Method: draw iid standard-normal shocks Z (paths x n_assets), transform
via Z_correlated = Z @ L.T where L = cholesky(corr_matrix) (lower
triangular), giving Z_correlated with unit variance per column and the
requested cross-asset correlation. Then apply the usual per-asset GBM
log-return transform to each column. This is the standard approach (e.g.
Glasserman 2004, "Monte Carlo Methods in Financial Engineering" ch.3) —
it is NOT the same as sampling the assets independently and then just
reporting a correlation on the output, which would silently do nothing.

Usage:
    python3 correlated_gbm.py --tickers NVDA,AMD,GOLD,SILVER \\
        --spots 130,140,2400,29 --mus 0.15,0.12,0.05,0.04 \\
        --sigmas 0.45,0.42,0.13,0.28 --days 5 \\
        --corr-csv corr_matrix.csv --weights 0.3,0.3,0.2,0.2
"""

import argparse
import json
import math
import numbers
import sys
from typing import Optional

import numpy as np

MAX_PATHS = 10_000_000


def validate_correlation_matrix(corr: np.ndarray, eps: float = 1e-8) -> None:
    """
    Raise ValueError if corr isn't a valid correlation matrix: symmetric,
    unit diagonal, entries in [-1, 1], and positive semi-definite (PSD).

    PSD-ness is checked via eigenvalues (np.linalg.eigvalsh), not by
    attempting a Cholesky decomposition — Cholesky requires strict
    positive-*definite*ness and rejects legitimate singular matrices, e.g.
    corr(A,B)=1.0 exactly (two tickers that track each other perfectly,
    like a stock/ADR pair), which is a valid but rank-deficient correlation
    structure with a near-zero smallest eigenvalue, not an impossible one.
    Sampling still needs a Cholesky factor; simulate_correlated_terminal_prices
    handles the singular-but-valid case separately with a small ridge term.
    """
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("correlation matrix must be square")
    if not np.allclose(corr, corr.T, atol=eps):
        raise ValueError("correlation matrix must be symmetric")
    if not np.allclose(np.diag(corr), 1.0, atol=eps):
        raise ValueError("correlation matrix diagonal must be exactly 1.0")
    if np.any(corr > 1 + eps) or np.any(corr < -1 - eps):
        raise ValueError("correlation matrix entries must be in [-1, 1]")
    min_eigenvalue = np.linalg.eigvalsh(corr).min()
    if min_eigenvalue < -eps:
        raise ValueError(
            f"correlation matrix is not positive semi-definite (smallest "
            f"eigenvalue {min_eigenvalue:.6g} < 0) — the pairwise "
            "correlations given cannot jointly hold (e.g. check for an "
            "inconsistent triple like corr(A,B)=0.9, corr(B,C)=0.9, "
            "corr(A,C)=-0.9, which is mathematically impossible)"
        )


def simulate_correlated_terminal_prices(
    spots: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    corr_matrix: np.ndarray,
    days: int,
    paths: int,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    trading_days_per_year: float = 252.0,
) -> np.ndarray:
    """
    Pure, vectorized correlated-GBM terminal-price simulation.

    spots, mus, sigmas: 1-D arrays of length n_assets (same conventions as
        gbm_stock_sim.simulate_terminal_prices — real-world drift, not
        risk-neutral; annualised sigma).
    corr_matrix: (n_assets, n_assets) correlation matrix of the assets'
        log-returns. This is a correlation, not covariance, matrix — the
        per-asset volatility is supplied separately via `sigmas` and
        applied after the correlated shocks are drawn.

    Returns an array of shape (paths, n_assets) of simulated terminal
    prices — one row per path, one column per asset, so a portfolio value
    or per-asset percentile can be computed by the caller with simple
    array ops.
    """
    if seed is not None and rng is not None:
        raise ValueError("seed and rng are mutually exclusive")
    spots = np.asarray(spots, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    corr_matrix = np.asarray(corr_matrix, dtype=float)

    n = spots.shape[0]
    if not (mus.shape == (n,) and sigmas.shape == (n,)):
        raise ValueError("spots, mus, sigmas must all be 1-D arrays of the same length")
    if n < 2:
        raise ValueError("need at least 2 assets — use gbm_stock_sim.py for a single asset")
    if corr_matrix.shape != (n, n):
        raise ValueError(f"corr_matrix must be shape ({n}, {n}) to match {n} assets")
    validate_correlation_matrix(corr_matrix)

    if not np.all(np.isfinite(spots)) or np.any(spots <= 0):
        raise ValueError("all spots must be finite and positive")
    if not np.all(np.isfinite(mus)):
        raise ValueError("all mus must be finite")
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas < 0):
        raise ValueError("all sigmas must be finite and non-negative")
    if not isinstance(days, numbers.Integral) or isinstance(days, bool) or days <= 0:
        raise ValueError("days must be a positive integer")
    if not isinstance(paths, numbers.Integral) or isinstance(paths, bool) or paths < 2:
        raise ValueError("paths must be an integer >= 2")
    if paths > MAX_PATHS:
        raise ValueError(f"paths must not exceed {MAX_PATHS}")
    if not math.isfinite(trading_days_per_year) or trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be a finite positive number")

    T = days / trading_days_per_year

    if rng is None:
        rng = np.random.default_rng(seed)

    try:
        L = np.linalg.cholesky(corr_matrix)  # lower-triangular; corr = L @ L.T
    except np.linalg.LinAlgError:
        # A valid (PSD) but singular correlation matrix, e.g. two assets
        # correlated at exactly 1.0, has no strict Cholesky factor — add a
        # small ridge so sampling can proceed. This changes the realized
        # correlation negligibly (ridge << any real correlation gap) but
        # makes the matrix strictly positive-definite.
        ridge = 1e-10
        print(
            f"note: correlation matrix is singular (valid but rank-deficient, "
            f"e.g. an exact 1.0 pairwise correlation) — adding a {ridge:.0e} "
            "ridge to the diagonal to allow sampling",
            file=sys.stderr,
        )
        L = np.linalg.cholesky(corr_matrix + ridge * np.eye(n))
    z_iid = rng.standard_normal((paths, n))
    z_correlated = z_iid @ L.T  # each column now unit-variance with the requested cross-correlation

    log_returns = (mus - 0.5 * sigmas**2) * T + sigmas * math.sqrt(T) * z_correlated
    terminal = spots * np.exp(log_returns)
    if not np.all(np.isfinite(terminal)):
        raise ValueError(
            "simulation produced non-finite terminal prices (overflow); "
            "reduce mu/sigma/days magnitude"
        )
    return terminal


def portfolio_value(terminal_prices: np.ndarray, spots: np.ndarray, weights: np.ndarray, portfolio_value_now: float) -> np.ndarray:
    """
    terminal_prices: (paths, n_assets). weights: fraction of portfolio
    value in each asset at t=0 (must sum to ~1). Returns simulated
    portfolio value per path — each asset's contribution scales by its own
    terminal/spot ratio, so weights need not equal share counts.
    """
    weights = np.asarray(weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f"weights must sum to 1.0 (got {weights.sum():.6f})")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative (no short positions supported here)")
    relative = terminal_prices / spots  # (paths, n_assets), each column's own return ratio
    return portfolio_value_now * (relative @ weights)


def _parse_float_list(s: str, name: str) -> np.ndarray:
    try:
        return np.array([float(x) for x in s.split(",")], dtype=float)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--{name} must be a comma-separated list of numbers") from None


def _load_corr_csv(path: str, n: int) -> np.ndarray:
    matrix = np.loadtxt(path, delimiter=",")
    if matrix.ndim != 2:
        raise ValueError(f"{path} must contain a 2-D matrix")
    if matrix.shape != (n, n):
        raise ValueError(f"{path} is {matrix.shape}, expected ({n}, {n}) to match the number of tickers")
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlated multi-asset GBM Monte Carlo simulation.")
    parser.add_argument("--tickers", required=True, help="Comma-separated ticker/asset labels")
    parser.add_argument("--spots", required=True, help="Comma-separated current spot prices, same order as --tickers")
    parser.add_argument("--mus", required=True, help="Comma-separated annualised drifts, same order as --tickers")
    parser.add_argument("--sigmas", required=True, help="Comma-separated annualised volatilities, same order as --tickers")
    parser.add_argument("--days", required=True, type=int, help="Horizon in trading days")
    parser.add_argument("--corr-csv", required=True, help="Path to an NxN CSV correlation matrix, same order as --tickers")
    parser.add_argument("--weights", default=None, help="Optional comma-separated portfolio weights (must sum to 1) for a portfolio-value summary")
    parser.add_argument("--portfolio-value", type=float, default=100.0, help="Portfolio notional value at t=0 if --weights given (default 100.0)")
    parser.add_argument("--paths", type=int, default=100_000, help="Number of simulation paths (default 100000)")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    parser.add_argument("--trading-days-per-year", type=float, default=252.0, help="Divisor for --days -> years (default 252)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.strip() for t in args.tickers.split(",")]
    if len(tickers) < 2 or any(not t for t in tickers):
        print("error: --tickers must list at least 2 non-empty, comma-separated labels", file=sys.stderr)
        raise SystemExit(1)
    n = len(tickers)

    try:
        spots = _parse_float_list(args.spots, "spots")
        mus = _parse_float_list(args.mus, "mus")
        sigmas = _parse_float_list(args.sigmas, "sigmas")
        for name, arr in (("spots", spots), ("mus", mus), ("sigmas", sigmas)):
            if len(arr) != n:
                raise ValueError(f"--{name} has {len(arr)} values, expected {n} to match --tickers")
        corr_matrix = _load_corr_csv(args.corr_csv, n)
        weights = _parse_float_list(args.weights, "weights") if args.weights else None
        if weights is not None and len(weights) != n:
            raise ValueError(f"--weights has {len(weights)} values, expected {n} to match --tickers")

        terminal = simulate_correlated_terminal_prices(
            spots, mus, sigmas, corr_matrix,
            days=args.days, paths=args.paths, seed=args.seed,
            trading_days_per_year=args.trading_days_per_year,
        )
    except (ValueError, argparse.ArgumentTypeError, OverflowError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    def pct(values, spot):
        median, p5, p95 = np.percentile(values, [50, 5, 95], method="linear")
        return {
            "median": float(median), "p5": float(p5), "p95": float(p95),
            "pct_change_median": float((median / spot - 1) * 100),
            "pct_change_p5": float((p5 / spot - 1) * 100),
            "pct_change_p95": float((p95 / spot - 1) * 100),
        }

    per_asset = {ticker: pct(terminal[:, i], spots[i]) for i, ticker in enumerate(tickers)}

    result = {
        "tickers": tickers, "spots": spots.tolist(), "mus": mus.tolist(), "sigmas": sigmas.tolist(),
        "days": args.days, "paths": args.paths, "seed": args.seed,
        "correlation_matrix": corr_matrix.tolist(),
        "per_asset": per_asset,
    }

    if weights is not None:
        try:
            port_values = portfolio_value(terminal, spots, weights, args.portfolio_value)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        median, p5, p95 = np.percentile(port_values, [50, 5, 95], method="linear")
        result["portfolio"] = {
            "weights": weights.tolist(),
            "value_now": args.portfolio_value,
            "median": float(median), "p5": float(p5), "p95": float(p95),
            "pct_change_median": float((median / args.portfolio_value - 1) * 100),
            "pct_change_p5": float((p5 / args.portfolio_value - 1) * 100),
            "pct_change_p95": float((p95 / args.portfolio_value - 1) * 100),
        }

    try:
        print(json.dumps(result, allow_nan=False, indent=2))
    except ValueError as exc:
        print(f"error: refusing to emit non-finite JSON output ({exc})", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
