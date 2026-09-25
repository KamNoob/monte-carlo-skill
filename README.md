# monte-carlo-skill

[Agent Skills](https://docs.claude.com/en/docs/claude-code/skills) for algorithmic technique selection — checklist-driven routing to the right approach, reviewed reference implementations, and validation guidance, instead of ad hoc "looks right, ran once" code.

## Monte Carlo simulation

- **[`monte-carlo/`](monte-carlo/README.md)** — the skill itself: `SKILL.md` (routing table + implementation template) and `references/` (one doc per technique).
- **[`monte-carlo-workspace/reference-scripts/`](monte-carlo-workspace/reference-scripts/README.md)** — the importable Python behind those references. Setup instructions and the module list are there.

Start with `monte-carlo/README.md` — it has a quick worked example and the full installation steps for both directories.

## Dynamic programming

- **[`dynamic-programming/`](dynamic-programming/README.md)** — the skill itself: `SKILL.md` (routing table + state-design recipe) and `references/` (one doc per problem family: knapsack, sequence alignment, interval DP, bitmask/graph DP, stochastic DP/MDPs, optimization techniques, pitfalls).
- **[`dynamic-programming-workspace/reference-scripts/`](dynamic-programming-workspace/reference-scripts/README.md)** — the importable, self-checking Python behind those references.

Start with `dynamic-programming/README.md`. The `stochastic-dp-mdp.md` reference is where the two skills connect: Longstaff-Schwartz American option pricing is dynamic programming's backward induction with Monte Carlo sampling standing in for an expectation too expensive to compute exactly.

## Research notes

- **[`research/`](research/dynamic-programming.md)** — background research notes on dynamic programming's theory and history, kept as source material for the skill above. The skill's `references/` are the actively-maintained, agent-facing documentation; this is the raw research they were built from.

## License

MIT — see [`LICENSE`](LICENSE).
