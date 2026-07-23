#!/usr/bin/env python3
"""
GBM stock-price Monte Carlo simulator.

Geometric Brownian motion models the price as a random walk with multiplicative
noise:
    dS_t / S_t = mu dt + sigma dW_t

Equivalently, the log-price follows a random walk with drift
(mu - 0.5*sigma**2) dt and diffusion sigma dW_t. Terminal prices are therefore
lognormally distributed. `mu` is the annualised
real-world expected drift and `sigma` is annualised volatility. This script uses
the real-world measure for finance check-ins, not the risk-neutral measure used
for option pricing.

Discretization notes:
  - Terminal prices are sampled directly from the exact GBM lognormal
    distribution, using one standard-normal draw per path.
  - T (years) is derived from --days via T = days / trading_days_per_year
    (default 252, the standard US-equities convention; override via
    --trading-days-per-year for other markets or 24/7 assets like crypto),
    and each path uses log-return (mu - 0.5*sigma**2) * T + sigma * sqrt(T) * Z.

Usage:
    python3 gbm_stock_sim.py --ticker NVDA --spot 130 --mu 0.15 --sigma 0.45 --days 5
"""

import argparse
import json
import math
import numbers
import sys
from typing import Optional

import numpy as np


MAX_PATHS = 10_000_000


def simulate_terminal_prices(
    spot: float,
    mu: float,
    sigma: float,
    days: int,
    paths: int,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    trading_days_per_year: float = 252.0,
) -> np.ndarray:
    """
    Pure, vectorized GBM terminal-price simulation. No I/O, no argparse,
    no printing — reusable by any caller (e.g. a weekly finance check-in
    script iterating over multiple tickers).

    Only the terminal price is needed, and under GBM it is exactly lognormal,
    so it is sampled directly with a single normal draw per path rather than
    simulating and summing intermediate daily steps.

    For multiple independent calls, create one `np.random.Generator`
    (e.g. `np.random.default_rng(seed)`) in the caller and pass it via `rng`
    on each call so the shock stream advances between tickers. `seed` is
    provided only as a convenience for single-call/CLI use.

    Returns an array of shape (paths,) of simulated terminal prices.
    """
    # For multiple calls, prefer a shared rng rather than reusing the same seed.
    if seed is not None and rng is not None:
        raise ValueError("seed and rng are mutually exclusive")
    if not isinstance(days, numbers.Integral) or isinstance(days, bool):
        raise ValueError("days must be an integer")
    if not isinstance(paths, numbers.Integral) or isinstance(paths, bool):
        raise ValueError("paths must be an integer")
    if not math.isfinite(spot) or spot <= 0:
        raise ValueError("spot must be a finite positive number")
    if not math.isfinite(mu):
        raise ValueError("mu must be finite")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be a finite non-negative number")
    if not math.isfinite(trading_days_per_year) or trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be a finite positive number")
    if days <= 0:
        raise ValueError("days must be positive")
    if paths < 2:
        raise ValueError("paths must be at least 2")
    if paths > MAX_PATHS:
        raise ValueError(f"paths must not exceed {MAX_PATHS}")

    T = days / trading_days_per_year  # horizon in years

    if rng is None:
        rng = np.random.default_rng(seed)

    # GBM terminal price is exactly lognormal, so draw a single normal shock
    # per path rather than simulating n_steps intermediate steps and summing
    # log-returns. This avoids an O(paths * n_steps) allocation (e.g. ~2GB
    # for paths=100_000, days=2520) with no change to the resulting
    # distribution, and matches the same corrected drift term
    # (mu - 0.5*sigma**2)*T + sigma*sqrt(T)*Z used previously.
    z = rng.standard_normal(paths)
    log_returns = (mu - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z

    terminal = spot * np.exp(log_returns)
    if not np.all(np.isfinite(terminal)):
        raise ValueError(
            "simulation produced non-finite terminal prices (overflow); "
            "reduce mu/sigma/days magnitude"
        )
    return terminal


def non_empty_ticker(value: str) -> str:
    ticker = value.strip()
    if not ticker:
        raise argparse.ArgumentTypeError("ticker must not be empty")
    return ticker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate GBM terminal stock prices and print a JSON summary."
    )
    parser.add_argument(
        "--ticker", required=True, type=non_empty_ticker, help="Ticker or asset label"
    )
    parser.add_argument("--spot", required=True, type=float, help="Current spot price")
    parser.add_argument("--mu", required=True, type=float, help="Annualised drift")
    parser.add_argument("--sigma", required=True, type=float, help="Annualised volatility")
    parser.add_argument("--days", required=True, type=int, help="Horizon in trading days")
    parser.add_argument(
        "--paths", type=int, default=100_000, help="Number of simulation paths (default 100000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility (default: random each run)",
    )
    parser.add_argument(
        "--trading-days-per-year",
        type=float,
        default=252.0,
        help=(
            "Divisor used to convert --days into years (default 252, the "
            "standard US-equities convention; use e.g. 365 for 24/7 assets "
            "like crypto)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        terminal_prices = simulate_terminal_prices(
            spot=args.spot,
            mu=args.mu,
            sigma=args.sigma,
            days=args.days,
            paths=args.paths,
            seed=args.seed,
            trading_days_per_year=args.trading_days_per_year,
        )
    except (ValueError, OverflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    # Percentiles use linear interpolation, not raw empirical order statistics.
    try:
        median, p5, p95 = np.percentile(
            terminal_prices, [50, 5, 95], method="linear"
        )
    except TypeError:
        median, p5, p95 = np.percentile(
            terminal_prices, [50, 5, 95], interpolation="linear"
        )
    mean = np.mean(terminal_prices)
    # Sample standard deviation (ddof=1) of simulated terminal prices.
    stdev = np.std(terminal_prices, ddof=1)

    if args.sigma > 0 and args.paths > 1 and mean == 0.0 and stdev == 0.0:
        print(
            "warning: all simulated terminal prices underflowed to exactly 0.0 "
            "(float64 cannot represent the true value at this mu/sigma/days "
            "magnitude) — result reflects numerical underflow, not a genuine "
            "zero-variance outcome; treat with caution",
            file=sys.stderr,
        )

    def pct_change(value: float) -> float:
        return (float(value) / args.spot - 1.0) * 100.0

    summary = {
        "ticker": args.ticker,
        "spot": args.spot,
        "mu": args.mu,
        "sigma": args.sigma,
        "days": args.days,
        "trading_days_per_year": args.trading_days_per_year,
        "paths": args.paths,
        "seed": args.seed,
        "mean": float(mean),
        "terminal_price_stdev_sample": float(stdev),
        "median": float(median),
        "p5": float(p5),
        "p95": float(p95),
        "pct_change_mean": pct_change(mean),
        "pct_change_median": pct_change(median),
        "pct_change_p5": pct_change(p5),
        "pct_change_p95": pct_change(p95),
    }
    try:
        print(json.dumps(summary, allow_nan=False))
    except ValueError as exc:
        print(f"error: refusing to emit non-finite JSON output ({exc})", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
