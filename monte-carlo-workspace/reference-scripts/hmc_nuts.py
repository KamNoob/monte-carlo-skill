#!/usr/bin/env python3
"""
Hamiltonian Monte Carlo (HMC) and the No-U-Turn Sampler (NUTS).

mcmc.md's Metropolis-Hastings sampler proposes a random-walk step and
accepts/rejects — for a correlated or high-dimensional posterior this
wastes most proposals rejecting or crawling, because a random-walk
proposal has no information about which directions are "downhill" in
log-probability. HMC uses the gradient of the log-posterior to simulate
a physical trajectory (treating -log p(theta) as a potential energy
surface, theta as a particle position) via the leapfrog integrator, then
does a single Metropolis correction on the whole trajectory. This moves
far in one step along the posterior's actual shape, not a random
direction, and its rejection rate is largely insensitive to correlation
between parameters (unlike M-H's).

NUTS removes the need to hand-pick a trajectory length L (too short:
random-walk-like behavior again; too long: wasted computation looping
back on itself) by doubling the trajectory forwards or backwards at
random until a "U-turn" is detected — the point where extending further
would start moving back toward where it started.

THIS IS A LEARNING/CORRECTNESS-REFERENCE IMPLEMENTATION, per the standing
guidance in mcmc.md: "For real Bayesian inference problems: recommend
PyMC (`pip install pymc`) or Stan — don't reimplement MCMC from scratch
unless the user explicitly wants to learn." Production NUTS (Stan, PyMC,
NumPyro) additionally uses: a memory-efficient O(log n) tree-building
variant instead of this naive O(n)-memory version (Hoffman & Gelman 2014,
Algorithm 3 vs. the Algorithm 2 implemented here), a mass matrix adapted
to the posterior's covariance (this implementation uses an identity mass
matrix throughout), multinomial rather than slice-based subtree
selection, and reverse-mode autodiff instead of numerical gradients.
Reach for a real library the moment the target's gradient isn't trivial
to hand-derive, or the problem has more than a handful of dimensions.

Reference: Hoffman & Gelman (2014), "The No-U-Turn Sampler: Adaptively
Setting Path Lengths in Hamiltonian Monte Carlo", JMLR 15.
Algorithm 2 (Naive NUTS) and Equation 6 (dual averaging) implemented here
match that paper's notation directly — variable names below (theta_minus,
theta_plus, r_minus, r_plus, s, j, mu, gamma, t0, kappa) are chosen to
match the paper so the code can be read alongside it.

Usage:
    python3 hmc_nuts.py --demo
"""

import argparse
import json
import math
import sys
from typing import Callable, Optional

import numpy as np


LogPGrad = Callable[[np.ndarray], tuple]  # theta -> (log_p, grad_log_p)


def numerical_grad(log_p: Callable[[np.ndarray], float], theta: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Central-difference gradient — a convenience for targets without an
    analytic gradient, NOT a substitute for autodiff. O(2*dim) log_p
    evaluations per gradient call, and accumulates float error; fine for
    the low-dimensional demo here, a real bottleneck/accuracy problem
    past a handful of dimensions. Prefer JAX/PyTorch autodiff or a
    hand-derived analytic gradient for anything beyond quick exploration.
    """
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = h
        grad[i] = (log_p(theta + e) - log_p(theta - e)) / (2 * h)
    return grad


def leapfrog(theta: np.ndarray, r: np.ndarray, grad_log_p: Callable[[np.ndarray], np.ndarray], eps: float, n_steps: int) -> tuple:
    """
    Standard leapfrog (Stormer-Verlet) integrator, symplectic (volume- and
    approximately energy-preserving), which is why HMC can propose distant
    states with high acceptance probability. Unit mass matrix throughout
    (kinetic energy = 0.5 * r.r) — a real posterior with very different
    per-parameter scales needs a diagonal or full mass matrix instead;
    not implemented here, see the module docstring.
    """
    r = r + 0.5 * eps * grad_log_p(theta)
    for i in range(n_steps):
        theta = theta + eps * r
        grad = grad_log_p(theta)
        r = r + (eps if i < n_steps - 1 else 0.5 * eps) * grad
    return theta, r


def hmc_step(theta: np.ndarray, log_p_grad: LogPGrad, eps: float, n_steps: int, rng: np.random.Generator) -> tuple:
    """
    One fixed-length HMC transition. Returns (new_theta, accept_prob).
    Hamiltonian H(theta,r) = -log_p(theta) + 0.5*r.r (potential + kinetic,
    unit mass) — accept with probability min(1, exp(H_current - H_proposed)),
    i.e. min(1, exp((log_p' - 0.5 r'.r') - (log_p - 0.5 r.r))).
    """
    log_p0, grad0 = log_p_grad(theta)
    r0 = rng.standard_normal(theta.shape)
    theta_new, r_new = leapfrog(theta, r0, lambda t: log_p_grad(t)[1], eps, n_steps)
    log_p_new, _ = log_p_grad(theta_new)
    if not (math.isfinite(log_p_new) and np.all(np.isfinite(theta_new))):
        return theta, 0.0  # divergence — reject, don't propagate inf/nan
    current_H = -log_p0 + 0.5 * r0 @ r0
    proposed_H = -log_p_new + 0.5 * r_new @ r_new
    accept_prob = min(1.0, math.exp(min(0.0, current_H - proposed_H)))
    if rng.uniform() < accept_prob:
        return theta_new, accept_prob
    return theta, accept_prob


def _dual_averaging_init(eps1: float) -> dict:
    return {"mu": math.log(10 * eps1), "log_eps_bar": 0.0, "H_bar": 0.0, "t": 0, "gamma": 0.05, "t0": 10.0, "kappa": 0.75}


def _dual_averaging_update(state: dict, H_t: float) -> float:
    """
    Nesterov dual averaging (Hoffman & Gelman 2014, Eq. 6), shrinking
    log(eps) toward mu = log(10*eps1). H_t = target_accept - actual_accept
    (the "how far above/below target was this step's acceptance" signal).
    Returns the new eps to use for the NEXT step. Called only during
    warmup — freeze eps at its running average once warmup ends, per the
    paper's "adapt during warmup, freeze after" standard practice.
    """
    state["t"] += 1
    t, gamma, t0, kappa, mu = state["t"], state["gamma"], state["t0"], state["kappa"], state["mu"]
    eta = t ** (-kappa)
    state["H_bar"] = (1 - 1.0 / (t + t0)) * state["H_bar"] + (1.0 / (t + t0)) * H_t
    log_eps = mu - (math.sqrt(t) / gamma) * state["H_bar"]
    state["log_eps_bar"] = eta * log_eps + (1 - eta) * state["log_eps_bar"]
    return math.exp(log_eps)


def run_hmc(theta0: np.ndarray, log_p_grad: LogPGrad, n_samples: int, n_warmup: int, n_steps: int = 20, target_accept: float = 0.65, eps_init: float = 0.1, seed: Optional[int] = None) -> dict:
    """
    Fixed-L HMC with dual-averaging step-size adaptation during warmup
    (frozen at the running average log_eps_bar afterward), following
    Hoffman & Gelman Algorithm 5's adaptation scheme applied to plain HMC.
    n_steps (L) is NOT adapted here — that's exactly the problem NUTS
    solves; see run_nuts below.
    """
    rng = np.random.default_rng(seed)
    theta = np.asarray(theta0, dtype=float).copy()
    da = _dual_averaging_init(eps_init)
    eps = eps_init
    samples = np.zeros((n_samples, len(theta)))
    accept_probs = np.zeros(n_samples)
    for i in range(n_warmup + n_samples):
        theta, accept_prob = hmc_step(theta, log_p_grad, eps, n_steps, rng)
        if i < n_warmup:
            eps = _dual_averaging_update(da, target_accept - accept_prob)
            if i == n_warmup - 1:
                eps = math.exp(da["log_eps_bar"])  # freeze immediately so sample 0 (next iteration) already uses it
        if i >= n_warmup:
            samples[i - n_warmup] = theta
            accept_probs[i - n_warmup] = accept_prob
    return {"samples": samples, "accept_probs": accept_probs, "final_eps": eps}


def _build_tree(theta: np.ndarray, r: np.ndarray, log_u: float, v: int, j: int, eps: float, log_p_grad: LogPGrad, theta0: np.ndarray, r0: np.ndarray, delta_max: float, rng: np.random.Generator) -> tuple:
    """
    Recursive BuildTree, Hoffman & Gelman Algorithm 2/6. Returns
    (theta_minus, r_minus, theta_plus, r_plus, theta_prime, C, s,
    alpha_sum, n_alpha, divergent) where C is the list of (theta,r) states
    in the slice, s is the continue-doubling indicator (0 = stop),
    alpha_sum/n_alpha accumulate the Metropolis-style acceptance statistic
    against the ORIGINAL (theta0, r0) for step-size adaptation (Algorithm
    6's alpha/n_alpha — accumulated WITHIN one BuildTree call's own
    recursion, per the paper; the caller must NOT further accumulate this
    across separate top-level doublings, see nuts_step), and divergent
    flags whether any leapfrog step in this subtree crossed the delta_max
    energy-error threshold (or produced a non-finite state).
    """
    if j == 0:
        theta_p, r_p = leapfrog(theta, r, lambda t: log_p_grad(t)[1], v * eps, 1)
        log_p_p, _ = log_p_grad(theta_p)
        joint = log_p_p - 0.5 * r_p @ r_p
        valid = math.isfinite(joint) and np.all(np.isfinite(theta_p))
        C = [(theta_p, r_p)] if valid and log_u <= joint else []
        s = 1 if (valid and joint > log_u - delta_max) else 0
        divergent = not valid or joint <= log_u - delta_max
        log_p0, _ = log_p_grad(theta0)
        joint0 = log_p0 - 0.5 * r0 @ r0
        alpha = min(1.0, math.exp(min(0.0, joint - joint0))) if valid else 0.0
        return theta_p, r_p, theta_p, r_p, theta_p, C, s, alpha, 1, divergent
    theta_minus, r_minus, theta_plus, r_plus, theta_prime, C, s, a_sum, n_a, div = _build_tree(theta, r, log_u, v, j - 1, eps, log_p_grad, theta0, r0, delta_max, rng)
    if s == 1:
        if v == -1:
            theta_minus, r_minus, _, _, theta_prime2, C2, s2, a_sum2, n_a2, div2 = _build_tree(theta_minus, r_minus, log_u, v, j - 1, eps, log_p_grad, theta0, r0, delta_max, rng)
        else:
            _, _, theta_plus, r_plus, theta_prime2, C2, s2, a_sum2, n_a2, div2 = _build_tree(theta_plus, r_plus, log_u, v, j - 1, eps, log_p_grad, theta0, r0, delta_max, rng)
        n_total = len(C) + len(C2)
        if n_total > 0 and rng.uniform() < len(C2) / n_total:
            theta_prime = theta_prime2
        s = s2 * (1 if (theta_plus - theta_minus) @ r_minus >= 0 else 0) * (1 if (theta_plus - theta_minus) @ r_plus >= 0 else 0)
        C = C + C2
        a_sum, n_a = a_sum + a_sum2, n_a + n_a2
        div = div or div2
    return theta_minus, r_minus, theta_plus, r_plus, theta_prime, C, s, a_sum, n_a, div


def nuts_step(theta: np.ndarray, log_p_grad: LogPGrad, eps: float, rng: np.random.Generator, max_treedepth: int = 10, delta_max: float = 1000.0) -> tuple:
    """One NUTS transition. Returns (new_theta, accept_stat_H, treedepth_reached, divergent)."""
    log_p0, _ = log_p_grad(theta)
    r0 = rng.standard_normal(theta.shape)
    joint0 = log_p0 - 0.5 * r0 @ r0
    log_u = joint0 - rng.exponential(1.0)  # log(Uniform(0, exp(joint0))) in distribution, avoids exponentiating joint0 directly

    theta_minus = theta_plus = theta
    r_minus = r_plus = r0
    j, theta_m, C, s = 0, theta, [(theta, r0)], 1
    a_sum, n_a = 0.0, 0  # from the CURRENT (most recent) doubling only, per Algorithm 6 — not summed across doublings
    divergent = False
    while s == 1 and j < max_treedepth:
        v = rng.choice([-1, 1])
        if v == -1:
            theta_minus, r_minus, _, _, theta_prime, C_prime, s_prime, a_sum, n_a, div = _build_tree(theta_minus, r_minus, log_u, v, j, eps, log_p_grad, theta, r0, delta_max, rng)
        else:
            _, _, theta_plus, r_plus, theta_prime, C_prime, s_prime, a_sum, n_a, div = _build_tree(theta_plus, r_plus, log_u, v, j, eps, log_p_grad, theta, r0, delta_max, rng)
        divergent = divergent or div
        if s_prime == 1 and C_prime:
            n_total = len(C) + len(C_prime)
            if rng.uniform() < len(C_prime) / n_total:
                theta_m = theta_prime
            C = C + C_prime
        s = s_prime * (1 if (theta_plus - theta_minus) @ r_minus >= 0 else 0) * (1 if (theta_plus - theta_minus) @ r_plus >= 0 else 0)
        j += 1
    accept_stat = a_sum / n_a if n_a > 0 else 0.0
    return theta_m, accept_stat, j, divergent


def run_nuts(theta0: np.ndarray, log_p_grad: LogPGrad, n_samples: int, n_warmup: int, target_accept: float = 0.8, eps_init: float = 0.1, max_treedepth: int = 10, seed: Optional[int] = None) -> dict:
    """
    NUTS with dual-averaging step-size adaptation during warmup. Default
    target_accept=0.8 matches Stan's practical default for NUTS (the
    paper's own experiments used values it doesn't fix as a hard
    recommendation; 0.8 is the widely-used value in production NUTS
    implementations for problems with non-trivial curvature).
    """
    rng = np.random.default_rng(seed)
    theta = np.asarray(theta0, dtype=float).copy()
    da = _dual_averaging_init(eps_init)
    eps = eps_init
    samples = np.zeros((n_samples, len(theta)))
    accept_stats = np.zeros(n_samples)
    treedepths = np.zeros(n_samples, dtype=int)
    divergences = np.zeros(n_samples, dtype=bool)
    for i in range(n_warmup + n_samples):
        theta, accept_stat, treedepth, divergent = nuts_step(theta, log_p_grad, eps, rng, max_treedepth=max_treedepth)
        if i < n_warmup:
            eps = _dual_averaging_update(da, target_accept - accept_stat)
            if i == n_warmup - 1:
                eps = math.exp(da["log_eps_bar"])  # freeze immediately so sample 0 (next iteration) already uses it
        if i >= n_warmup:
            samples[i - n_warmup] = theta
            accept_stats[i - n_warmup] = accept_stat
            treedepths[i - n_warmup] = treedepth
            divergences[i - n_warmup] = divergent
    return {"samples": samples, "accept_stats": accept_stats, "treedepths": treedepths, "divergences": divergences, "final_eps": eps}


def _demo() -> dict:
    """
    Target: a 2-D correlated Gaussian (rho=0.9) — exactly the case where
    random-walk Metropolis-Hastings struggles (must take small steps to
    keep a reasonable acceptance rate along the narrow correlated ridge)
    but HMC/NUTS, using the gradient, do not.
    """
    mean = np.array([1.0, -2.0])
    cov = np.array([[1.0, 0.9], [0.9, 1.0]])
    cov_inv = np.linalg.inv(cov)

    def log_p_grad(theta):
        d = theta - mean
        log_p = -0.5 * d @ cov_inv @ d
        grad = -cov_inv @ d
        return log_p, grad

    # HMC warmup is longer than NUTS's: fixed-L HMC's accept-rate-vs-eps
    # curve has a hard stability cliff (leapfrog diverges outright above
    # some eps, not a smooth falloff — see hmc-nuts.md's "Stability cliff"
    # section for the measured curve), which makes dual-averaging converge
    # more slowly/noisily than it does for NUTS on the same target.
    hmc = run_hmc(np.zeros(2), log_p_grad, n_samples=5000, n_warmup=4000, n_steps=20, seed=1)
    nuts = run_nuts(np.zeros(2), log_p_grad, n_samples=5000, n_warmup=1000, seed=2)

    def summarize(samples):
        return {
            "mean": samples.mean(axis=0).tolist(),
            "cov": np.cov(samples.T).tolist(),
        }

    return {
        "true_mean": mean.tolist(),
        "true_cov": cov.tolist(),
        "hmc": {**summarize(hmc["samples"]), "mean_accept_prob": float(hmc["accept_probs"].mean()), "final_eps": hmc["final_eps"]},
        "nuts": {**summarize(nuts["samples"]), "mean_accept_stat": float(nuts["accept_stats"].mean()), "mean_treedepth": float(nuts["treedepths"].mean()), "final_eps": nuts["final_eps"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HMC/NUTS reference sampler demo on a correlated 2-D Gaussian.")
    parser.add_argument("--demo", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(_demo(), indent=2))


if __name__ == "__main__":
    main()
