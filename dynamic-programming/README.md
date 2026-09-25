# Dynamic Programming Skill

Dynamic programming (DP) solves a problem by breaking it into overlapping subproblems, solving each one **once**, and reusing the stored result instead of recomputing it. It applies whenever two things hold: **optimal substructure** (the best solution is built from the best solutions to its subproblems — Bellman's Principle of Optimality) and **overlapping subproblems** (the same subproblems recur, so there's something worth caching). Ask an LLM for a DP solution and the recurrence is usually roughly right — the bugs that slip through are quieter than that: a knapsack capacity loop iterating the wrong direction (silently allowing an item to be reused), a "count the ways" DP with the loop nesting swapped (silently counting permutations instead of combinations), a state missing one piece of information the answer actually depends on. None of these crash; they just return a plausible, wrong number.

This is an [Agent Skill](https://docs.claude.com/en/docs/claude-code/skills) that gives an AI coding agent a routing table from problem type to technique, a state-design recipe, working reference implementations for the classic DP families, and a validation checklist — including the specific loop-direction and loop-nesting gotchas above.

## A bit of history

Richard Bellman developed dynamic programming at the RAND Corporation starting in 1949. "Programming" meant planning/scheduling (as in "linear programming"), not code; Bellman reportedly chose "dynamic" partly to keep the word "research" out of the name, since Secretary of Defense Charles Wilson had a well-known aversion to it. The Principle of Optimality he formulated — *"an optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision"* — is still the exact justification for why every recurrence in this skill is valid.

## What's here

- **`SKILL.md`** — the entry point. A routing table maps problem types (1-D sequential, knapsack, sequence alignment, interval, bitmask, graph shortest-path, stochastic/MDP) to the right reference, plus a state-design recipe, a top-down-vs-bottom-up guide, and a validation checklist.
- **`references/`** — one markdown file per problem family, each with the recurrence, the reasoning behind it, the specific bugs that tend to slip through, and a pointer to a reviewed, self-checking reference script:
  - `linear-and-knapsack-dp.md` — 1-D sequential DP (Fibonacci-shaped, house robber, Kadane) and the knapsack family (0/1 knapsack, subset sum, partition)
  - `sequence-alignment.md` — LCS, edit distance, and the space-vs-reconstruction tradeoff
  - `interval-dp.md` — matrix-chain multiplication, palindrome partitioning
  - `bitmask-and-graph-dp.md` — Held-Karp TSP, Bellman-Ford, Floyd-Warshall
  - `stochastic-dp-mdp.md` — MDPs, the Bellman equation, value/policy iteration, the curse of dimensionality, and where Monte Carlo sampling (Longstaff-Schwartz American option pricing) takes over when the state space is too large to tabulate
  - `optimization-techniques.md` — rolling arrays, Hirschberg's algorithm, monotone queues, convex hull trick, Knuth/D&C optimization, matrix exponentiation
  - `pitfalls-and-checklist.md` — the specific bugs above in full, DP vs. greedy vs. divide-and-conquer vs. backtracking, and a quick "is this DP?" checklist
- **Companion reference scripts**: `../dynamic-programming-workspace/reference-scripts/` — runnable, self-checking Python behind every technique above (see that directory's README for setup and the module list).

## Installation

Two parts, installed the same way as this repo's `monte-carlo` skill: the skill itself (`dynamic-programming/`) and its companion scripts (`dynamic-programming-workspace/`).

```bash
git clone <this-repo-url> /tmp/dp-skill
cp -r /tmp/dp-skill/dynamic-programming ~/.claude/skills/
cp -r /tmp/dp-skill/dynamic-programming-workspace ~/.claude/skills/
```

For Codex CLI, use `~/.codex/skills/` instead. Keep the two directories as siblings — `references/*.md` files point at `../dynamic-programming-workspace/reference-scripts/*.py` by relative path.

Most reference scripts are pure standard-library Python and need no setup. Two (`mdp_value_iteration.py`, `lsm_option_pricing.py`) need `numpy` — reuse the venv from this repo's `monte-carlo-workspace/reference-scripts/` if you have it installed alongside, rather than creating a second one:

```bash
cd ~/.claude/skills/dynamic-programming-workspace/reference-scripts
.venv/bin/python3 knapsack.py   # or point at an existing shared venv with numpy
```

## License

MIT — see `LICENSE`.
