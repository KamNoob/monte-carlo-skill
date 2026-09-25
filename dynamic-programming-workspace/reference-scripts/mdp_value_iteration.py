"""Stochastic DP: value iteration and policy iteration for finite MDPs.

State: value function V(s). Transition: the Bellman optimality equation

    V*(s) = max_a [ R(s,a) + gamma * sum_s' P(s'|s,a) * V*(s') ]

P is a list of S x S matrices, one per action (P[a][s, s'] = transition
probability). R is an S x A matrix of expected immediate rewards.

Value iteration applies the Bellman operator repeatedly; it's a
gamma-contraction in sup-norm, so it converges geometrically to the unique
fixed point regardless of the starting V. Policy iteration alternates
exact policy evaluation (solving a linear system) with greedy policy
improvement; it usually needs far fewer iterations but each one is more
expensive (an S x S linear solve).
"""
from __future__ import annotations
import numpy as np


def value_iteration(P: list[np.ndarray], R: np.ndarray, gamma: float = 0.9,
                     tol: float = 1e-10, max_iter: int = 100_000):
    """Returns (V*, greedy_policy) where policy[s] is the best action index."""
    n_states = R.shape[0]
    V = np.zeros(n_states)
    for _ in range(max_iter):
        Q = R + gamma * np.stack([P[a] @ V for a in range(len(P))], axis=1)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, Q.argmax(axis=1)
        V = V_new
    return V, Q.argmax(axis=1)


def policy_evaluation(policy: np.ndarray, P: list[np.ndarray], R: np.ndarray,
                       gamma: float = 0.9) -> np.ndarray:
    """Exact V^pi by solving the linear system (I - gamma*P_pi) V = R_pi."""
    n_states = R.shape[0]
    P_pi = np.stack([P[policy[s]][s] for s in range(n_states)])
    R_pi = np.array([R[s, policy[s]] for s in range(n_states)])
    return np.linalg.solve(np.eye(n_states) - gamma * P_pi, R_pi)


def policy_iteration(P: list[np.ndarray], R: np.ndarray, gamma: float = 0.9,
                      max_iter: int = 10_000):
    n_states, n_actions = R.shape
    policy = np.zeros(n_states, dtype=int)
    for _ in range(max_iter):
        V = policy_evaluation(policy, P, R, gamma)
        Q = R + gamma * np.stack([P[a] @ V for a in range(n_actions)], axis=1)
        new_policy = Q.argmax(axis=1)
        if np.array_equal(new_policy, policy):
            return V, policy
        policy = new_policy
    return V, policy


if __name__ == "__main__":
    # Two-state MDP: action 0 = stay, action 1 = switch. Reward 1 only in
    # state 1, 0 in state 0. Optimal policy: switch out of state 0 (once),
    # then stay in state 1 forever, collecting reward 1 every step.
    P = [np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]])]
    R = np.array([[0.0, 0.0], [1.0, 1.0]])
    gamma = 0.9

    V_vi, pi_vi = value_iteration(P, R, gamma)
    # Closed form: V(1) = 1/(1-gamma) = 10; V(0) = gamma*V(1) = 9 (one switch, then stay)
    assert np.allclose(V_vi, [9.0, 10.0]), V_vi
    assert list(pi_vi) == [1, 0]  # switch from state 0, stay in state 1

    V_pi, pi_pi = policy_iteration(P, R, gamma)
    assert np.allclose(V_pi, V_vi, atol=1e-8)
    assert list(pi_pi) == list(pi_vi)

    # A slightly larger 3-state stochastic case, cross-checked between the
    # two methods (they must agree since both solve the same Bellman equation).
    rng = np.random.default_rng(0)
    n = 3
    P3 = []
    for _ in range(2):
        M = rng.dirichlet(np.ones(n), size=n)  # random row-stochastic matrix
        P3.append(M)
    R3 = rng.uniform(0, 1, size=(n, 2))
    Vv, pv = value_iteration(P3, R3, gamma=0.95)
    Vp, pp = policy_iteration(P3, R3, gamma=0.95)
    assert np.allclose(Vv, Vp, atol=1e-6), (Vv, Vp)
    assert list(pv) == list(pp)
    print("mdp_value_iteration: all checks passed")
