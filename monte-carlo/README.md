# Monte Carlo Simulation Skill

Ask an LLM to "run a Monte Carlo simulation" for something and you'll usually get code back fast — and it'll often be subtly wrong: a lognormal sample that's silently biased, a confidence interval formula that only works near p=0.5, a scipy distribution fit that quietly collapsed to a boundary value. None of that shows up unless you know to check for it.

This is an [Agent Skill](https://docs.claude.com/en/docs/claude-code/skills) that gives an AI coding agent (Claude Code, Codex, or anything else that reads plain markdown) a checklist for doing this properly: figure out what kind of problem it actually is, pick the right technique, implement it against a reviewed reference, and validate the result against something you can check independently before trusting it. Every reference script in here started as "looks right, ran once" and had at least one real bug found by adversarial review before it earned a place in the repo — see the bottom of this README for what that turned up.

## Quick example

The reference scripts are plain, dependency-light Python — usable standalone, not just through an agent:

```bash
$ .venv/bin/python3 monte-carlo-workspace/reference-scripts/gbm_stock_sim.py \
    --ticker DEMO --spot 100 --mu 0.08 --sigma 0.30 --days 20 --seed 42
{"ticker": "DEMO", "spot": 100.0, "mu": 0.08, "sigma": 0.3, "days": 20,
 "mean": 100.6, "median": 100.2, "p5": 87.16, "p95": 115.25, ...}
```

100,000 simulated paths of a $100 stock over 20 trading days at 8% drift / 30% annualised volatility, reproducibly seeded — 90% of outcomes land between $87 and $115. That's the terminal-price shortcut (`gbm_stock_sim.py`'s docstring explains why sampling the terminal lognormal directly, instead of stepping through the whole path, is both faster and exactly correct when only the endpoint matters). The other 8 modules follow the same pattern for harder problems: fat tails, portfolio correlation, Bayesian posteriors, hidden-state tracking.

## What's here

- **`SKILL.md`** — the entry point. A routing table maps problem types (estimation, financial, MCMC/Bayesian, extreme-tail risk, hidden-state tracking, etc.) to the right technique and reference doc, plus a standard implementation template, common pitfalls, and validation checklist.
- **`references/`** — one markdown file per technique, each with the underlying math, when to use it vs. alternatives, known gotchas (scipy parameterization traps, boundary-value failure modes, etc.), and a pointer to a reviewed reference script:
  - `financial.md`, `correlated-assets.md` — GBM, options, single- and multi-asset portfolio VaR/CVaR
  - `stable-distributions.md`, `variance-gamma.md`, `extreme-value-theory.md` — fat-tailed / deep-tail risk modeling beyond Gaussian assumptions
  - `copulas.md` — dependence structure (Gaussian/t/Clayton) independent of marginal choice
  - `mcmc.md`, `hmc-nuts.md` — Bayesian posterior sampling, Metropolis-Hastings through NUTS
  - `particle-filters.md` — hidden-state tracking in nonlinear/non-Gaussian state-space models
  - `time-series.md` — ARIMA/SARIMA + residual-bootstrap forecasting
  - `variance-reduction.md` — antithetic/control variates, importance sampling, stratified/QMC
  - `parameterization-conventions.md` — numpy vs. scipy distribution parameter gotchas
- **Companion reference scripts**: `../monte-carlo-workspace/reference-scripts/` — importable, adversarially-reviewed Python implementations backing the techniques above (see that directory's own README for setup and the list of modules).

## Installation

This repo has two parts that install separately: the skill itself (`monte-carlo/`) and its companion Python scripts (`monte-carlo-workspace/reference-scripts/`).

### 1. Install the skill

Copy both directories into your agent's skill folder — same layout, either tool:

```bash
git clone <this-repo-url> /tmp/mc-skill
cp -r /tmp/mc-skill/monte-carlo ~/.claude/skills/
cp -r /tmp/mc-skill/monte-carlo-workspace ~/.claude/skills/
```

For Codex CLI, use `~/.codex/skills/` instead. The `description` field in `SKILL.md`'s frontmatter is what the agent uses to decide when to load it — no further registration step for either tool.

If the agent's skill folder lives somewhere else, adjust the destination path accordingly; the two directories should stay siblings (`references/*.md` files point at `../monte-carlo-workspace/reference-scripts/*.py` by relative path).

### 2. Set up the Python environment for the reference scripts

```bash
cd ~/.claude/skills/monte-carlo-workspace/reference-scripts
python3 -m venv .venv
.venv/bin/pip install numpy scipy matplotlib pandas statsmodels
```

Verify it worked:

```bash
.venv/bin/python3 -c "import numpy, scipy; print('ok')"
```

See that directory's own `README.md` for the module list and what each script does.

### Notes

- No dependency on Claude specifically — everything here is plain markdown and Python. What's tool-specific is only the discovery/auto-loading mechanism (the `SKILL.md` frontmatter + whatever skill-routing the agent uses); the knowledge and scripts are just files any agent or human can read and run directly.
- If you already have a shared venv you use for other Python tooling on the same machine, point the reference-scripts docs at that instead of creating a second one — don't proliferate venvs per tool.

## What "adversarially reviewed" actually caught

A sample of the real bugs found and fixed before these scripts were trusted (full list per-module in each `references/*.md`):

- `stable_distribution.py`: scipy's MLE fit silently converges to a boundary value (alpha=2, beta=-1) that looks like a valid fit but isn't a real optimum — now rejected explicitly.
- `correlated_gbm.py`: PSD validation via `np.linalg.cholesky` was wrongly rejecting *valid* singular correlation matrices (e.g. a stock/ADR pair correlated at exactly 1.0) — switched to `eigvalsh`.
- `hmc_nuts.py`: NUTS's step-size adaptation was summing the accept-stat across every doubling instead of using only the top-level call, biasing step-size tuning in a way that never showed up in mean/covariance/ESS checks.
- `extreme_value.py`: the sign of a fitted tail-shape parameter can flip at only 25 exceedances — raised the floor and added bootstrap confidence intervals so a caller sees real precision instead of a falsely-confident point estimate.
- `particle_filter.py`: particles were transitioned before being weighted against the first observation, contradicting the model's own documented contract — a ~2% bias in a case that looked fine at a glance.

None of these were visible from reading the code once. They came out under a review pass that specifically distrusts "ran without crashing" as evidence of correctness.

## License

MIT — see `LICENSE`.
