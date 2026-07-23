# Correlated Multi-Asset GBM Reference

## When to use this instead of independent per-asset GBM

`references/financial.md`'s `simulate_gbm_paths` and `gbm_stock_sim.py` model
one asset at a time with its own independent shock. That's the right tool
when the question is genuinely single-asset ("where might NVDA land in a
week"). It's the wrong tool the moment the question is about a **portfolio**
of assets that move together — NVDA and AMD share semiconductor-sector risk,
gold and silver share precious-metals risk. Sampling them independently
understates the odds of a joint drawdown: a portfolio VaR computed from
independent per-asset sims will be too optimistic whenever the real assets
are positively correlated, because it's implicitly assuming diversification
that isn't really there.

Use `correlated_gbm.py` when:
- The user asks about portfolio-level risk (VaR/CVaR, "how bad could my
  whole position get") across more than one asset
- The assets have an obvious economic reason to move together (same
  sector, same macro driver, same underlying commodity)
- A prior independent-per-asset sim's tail estimates look too tight
  compared to how the assets have actually historically moved together

Keep using single-asset `gbm_stock_sim.py` when there's only one asset in
play, or when a multi-asset question genuinely has no correlation
assumption behind it — e.g. reporting several unrelated single-asset
forecasts side by side, with no combined portfolio figure requested.

## Method

Draw iid standard-normal shocks `Z` of shape `(paths, n_assets)`, transform
via `Z_correlated = Z @ L.T` where `L = np.linalg.cholesky(corr_matrix)`
(lower triangular, `corr = L @ L.T`). Each column of `Z_correlated` still
has unit variance, but now the requested cross-asset correlation. Apply
the usual per-asset GBM log-return transform to each column using that
asset's own `mu`/`sigma`. This is the standard approach (Glasserman 2004,
*Monte Carlo Methods in Financial Engineering*, ch. 3) — verified
empirically in the reference script: sampling 500k paths from a 4-asset
correlation matrix with 0.7 and 0.85 off-diagonal entries recovered
realized sample correlations of 0.699 and 0.850, and the identity-matrix
(fully independent) case recovered ~0 cross-correlation.

**Correlation matrix, not covariance matrix.** Per-asset volatility is
supplied separately via `sigmas` and applied after the correlated shocks
are drawn — don't pass a covariance matrix here, its diagonal wouldn't be
1.0 and the validation would (correctly) reject it.

## Validity gotcha: not every matrix of pairwise correlations is realizable

A correlation matrix must be positive semi-definite to admit a Cholesky
decomposition — and not every set of pairwise correlations that individually
look plausible actually is jointly consistent. Classic example: corr(A,B)=0.9,
corr(B,C)=0.9, corr(A,C)=-0.9 is impossible (if A and B move almost
identically, and B and C move almost identically, A and C can't be
near-perfect opposites). The reference script's `validate_correlation_matrix`
checks this by attempting the Cholesky decomposition itself, not just
checking symmetry/diagonal/range — those three checks alone would pass the
impossible-triple example above.

If real historical correlations from data (rather than hand-specified
ones) are being used, this is rarely an issue — an empirical correlation
matrix computed from actual return data is guaranteed positive
semi-definite. The failure mode is almost always hand-typed or
partially-guessed correlation values.

**Singular-but-valid matrices are a separate case from invalid ones.** A
matrix can be genuinely PSD yet singular — e.g. two tickers given an exact
1.0 correlation (a stock/ADR pair, or a typo'd duplicate ticker) — and
`np.linalg.cholesky` rejects that too, even though it's mathematically
valid, because Cholesky requires strict positive-*definite*ness, not just
PSD. The reference script checks PSD-ness via eigenvalues (not Cholesky)
so it doesn't misdiagnose this case as an impossible-triple problem, and
separately adds a tiny (1e-10) ridge to the diagonal before the sampling
Cholesky if the exact matrix turns out to be singular — negligible effect
on the realized correlation, but lets sampling proceed instead of failing
on a legitimate input.

## Reviewed reference implementation

`../../monte-carlo-workspace/reference-scripts/correlated_gbm.py`
— takes per-asset spot/mu/sigma plus an NxN correlation matrix CSV, runs
the correlated GBM simulation, reports per-asset percentiles, and
optionally a portfolio-value distribution given weights (`relative return
per asset @ weights`, so weights don't need to equal share counts).

## Validation

- **Realized-correlation check**: for any new correlation matrix, sample a
  few hundred thousand paths and confirm `np.corrcoef` on the simulated
  log-returns recovers the input matrix (as done above) — this is a cheap,
  concrete sanity check that the Cholesky wiring wasn't accidentally
  transposed or applied to the wrong axis.
- **Independent-case regression**: an identity correlation matrix should
  recover ~0 realized cross-correlation and match `gbm_stock_sim.py`'s
  independent per-asset results for the same spot/mu/sigma/days/seed —
  useful as a quick "did I break the single-asset case" check.
- Standard GBM validation otherwise applies (see `SKILL.md` Step 4) —
  convergence vs N, coverage if used for forecasting.
