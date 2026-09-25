# Dynamic Programming (DP)

Research notes, saved as long-term reference. Every code example below is
reproduced in `dynamic_programming_examples.py` next to this file, and that
file asserts the stated results (run it with any Python that has `numpy`).

## 1. What it is

Dynamic programming solves a problem by breaking it into overlapping
subproblems, solving each subproblem **once**, storing the result, and
building the final answer from the stored results. It works only when two
properties hold:

| Property | Meaning | If it's missing |
|---|---|---|
| **Optimal substructure** | An optimal solution contains optimal solutions to its subproblems | DP gives wrong answers (e.g. longest *simple* path in a general graph) |
| **Overlapping subproblems** | The same subproblems recur many times | DP is correct but no faster than plain divide-and-conquer (e.g. merge sort) |

**Bellman's Principle of Optimality** (the formal version of optimal
substructure): *"an optimal policy has the property that whatever the
initial state and initial decision are, the remaining decisions must
constitute an optimal policy with regard to the state resulting from the
first decision."*

**History.** Richard Bellman developed DP at the RAND Corporation from
1949 into the 1950s for multistage decision processes. "Programming" meant
planning/scheduling, as in "linear programming", not coding. Bellman said he
picked "dynamic" partly because it sounded impressive and hard to object
to. He was shielding mathematical research from Secretary of Defense
Charles Wilson, who disliked the word "research".

## 2. The recipe (how to design a DP)

1. **Define the state.** Choose the smallest set of parameters that
   determines the answer to a subproblem (e.g. `dp[i][j]` = edit distance
   between the first `i` chars of `a` and the first `j` chars of `b`).
   Choosing the state is where most of the difficulty lies.
2. **Write the recurrence (transition).** Express `dp[state]` in terms of
   smaller states. For optimization problems this is a `min` or `max` over
   the available choices.
3. **Base cases.** The trivial states, such as empty string, zero capacity
   or the terminal time.
4. **Evaluation order.** Every dependency must be computed before the
   states that use it (topological order of the subproblem DAG).
5. **Answer location.** Say which state or combination of states is the
   final answer, and whether you also need to *reconstruct* the choices
   (store argmax/back-pointers).
6. **Complexity** is roughly (# states) × (work per transition).
   Then look for space savings: if each row only depends on the previous
   row, keep only 1–2 rows.

## 3. Two implementation styles

| | Top-down (memoization) | Bottom-up (tabulation) |
|---|---|---|
| How | Recursion + cache | Iterative loops filling a table |
| Computes | Only the states actually reached | Every state |
| Pros | Direct translation of the recurrence; skips unreachable states | No recursion limit, less call overhead, easy space optimization |
| Cons | Python recursion limit (~1000 default); overhead of cache | Must work out evaluation order yourself |

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_memo(n):
    return n if n < 2 else fib_memo(n - 1) + fib_memo(n - 2)

def fib_tab(n):                 # O(n) time, O(1) space
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```

Naive recursive Fibonacci is O(φⁿ); either DP version is O(n).

## 4. Canonical problem families

| Family | State | Example problems | Typical complexity |
|---|---|---|---|
| **1-D linear** | `dp[i]` = best using the first `i` items | Fibonacci, climbing stairs, house robber, max subarray (Kadane), coin change | O(n) – O(n·k) |
| **Knapsack / subset** | `dp[i][capacity]` | 0/1 knapsack, subset sum, partition equal subset | O(n·W), pseudo-polynomial |
| **Two-sequence** | `dp[i][j]` over prefixes | LCS, edit distance, sequence alignment (Needleman–Wunsch, Smith–Waterman) | O(m·n) |
| **Interval** | `dp[i][j]` over substrings/ranges | Matrix-chain multiplication, optimal BST, burst balloons, palindrome partitioning | O(n³) |
| **Grid** | `dp[r][c]` | Unique paths, min path sum | O(R·C) |
| **Tree DP** | value per subtree, computed post-order | Max independent set on a tree, tree diameter | O(n) |
| **Bitmask DP** | `dp[mask][i]` over subsets | Travelling Salesman (Held–Karp), assignment | O(2ⁿ·n²) |
| **Digit DP** | position × tight-flag × aggregate | Count numbers ≤ N with a digit property | O(digits · states) |
| **Graph shortest paths** | distances relaxed over a DAG / edges | Bellman–Ford, Floyd–Warshall, shortest path in a DAG | O(VE), O(V³) |
| **Sequential decisions under uncertainty** | value function `V(state, t)` | MDPs, optimal stopping, inventory control, American options | see §6 |

### 0/1 knapsack, O(n·W) time and O(W) space

```python
def knapsack(weights, values, W):
    dp = [0] * (W + 1)
    for w, v in zip(weights, values):
        for c in range(W, w - 1, -1):   # iterate DOWN so each item is used at most once
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[W]
# knapsack([1,3,4,5], [1,4,5,7], 7) == 9
```

Iterating capacity **upward** turns this into the *unbounded* knapsack
(each item reusable). This loop direction is a common source of bugs.

### Edit distance (Levenshtein)

```python
def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = a[i - 1] != b[j - 1]
            dp[i][j] = min(dp[i - 1][j] + 1,        # delete
                           dp[i][j - 1] + 1,        # insert
                           dp[i - 1][j - 1] + cost) # substitute / match
    return dp[m][n]
# edit_distance("kitten", "sitting") == 3
```

### Longest increasing subsequence

The textbook DP is O(n²) (`dp[i] = 1 + max dp[j] for j<i, a[j]<a[i]`). The
"patience sorting" refinement uses binary search to reach O(n log n):

```python
import bisect
def lis_length(xs):
    tails = []                       # tails[k] = smallest tail of an increasing subsequence of length k+1
    for x in xs:
        i = bisect.bisect_left(tails, x)
        tails[i:i + 1] = [x]
    return len(tails)
# lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4
```

## 5. Optimization techniques (when the plain DP is too slow)

- **Rolling arrays.** Keep only the rows you still need. O(m·n) space
  becomes O(n).
- **Hirschberg's algorithm.** Reconstructs an LCS or alignment in linear
  space using divide and conquer.
- **Monotone queue / deque.** Speeds up sliding-window `min`/`max`
  transitions.
- **Convex hull trick / Li Chao tree.** Handles transitions of the form
  `min_j (m_j·x + b_j)`, bringing O(n²) down to O(n log n) or O(n).
- **Divide & conquer optimization, Knuth's optimization.** Apply when the
  optimal split point is monotone (quadrangle inequality). Knuth brings
  some interval DPs from O(n³) to O(n²).
- **Meet-in-the-middle.** Splits an exponential state space in half.
- **Matrix exponentiation.** Solves linear recurrences in O(k³ log n)
  (e.g. Fibonacci in O(log n)).

## 6. Stochastic DP: MDPs, the Bellman equation, and the link to Monte Carlo

For decisions under uncertainty, the DP state is a *value function*. For a
Markov Decision Process with states `s`, actions `a`, transition
`P(s'|s,a)`, reward `R(s,a)` and discount `γ < 1`:

```
V*(s) = max_a [ R(s,a) + γ · Σ_{s'} P(s'|s,a) · V*(s') ]      (Bellman optimality equation)
```

- **Finite horizon / optimal stopping.** Use backward induction from the
  terminal time: `V_T` is known, then compute `V_{T-1}`, …, `V_0`.
- **Infinite horizon.** Use **value iteration**, which applies the Bellman
  operator until it converges. It is a γ-contraction, so it converges
  geometrically. Alternatively use **policy iteration**: evaluate the
  current policy, improve it greedily, and repeat. It usually needs fewer
  but more expensive iterations.

```python
import numpy as np
def value_iteration(P, R, gamma=0.9, tol=1e-10):   # P[a] is an S×S matrix, R is S×A
    V = np.zeros(R.shape[0])
    while True:
        Q = R + gamma * np.stack([P[a] @ V for a in range(len(P))], axis=1)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, Q.argmax(axis=1)
        V = V_new
```

**The curse of dimensionality** (Bellman's own phrase): the state space
grows exponentially with the number of state variables, so exact tabular
DP becomes infeasible. The response is **Approximate DP (ADP) /
reinforcement learning**. You replace the table with a function
approximator, and replace the exact expectation `Σ P(s'|s,a)·V(s')` with
**Monte Carlo samples**:

| Method | DP component replaced by sampling |
|---|---|
| Monte Carlo policy evaluation | V(s) ≈ average sampled return |
| TD learning / Q-learning | One-sample Bellman backup: `Q ← Q + α(r + γ max Q' − Q)` |
| Fitted value iteration / fitted Q | Regression of sampled backups onto features |
| **Longstaff–Schwartz (LSM)** | Regression of discounted future cash flows onto basis functions of the state |

### Worked example: American put = optimal stopping = DP

At each exercise date the holder compares exercising now against
continuing: `V_t = max(payoff_t, E[disc · V_{t+1} | S_t])`. On a binomial
lattice that conditional expectation is exact, which gives backward
induction. With simulated paths it is unknown, and Longstaff–Schwartz
(2001, *Review of Financial Studies* 14:113–147) estimate it with a
cross-sectional least-squares regression on in-the-money paths. LSM is now
the standard Monte Carlo method for American and Bermudan options,
especially path-dependent or multi-factor ones where lattices and finite
differences break down.

```python
def american_put_lsm(S0, K, r, sigma, T, steps=50, n_paths=100_000, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / steps
    Z = rng.standard_normal((n_paths // 2, steps)); Z = np.vstack([Z, -Z])   # antithetic
    S = np.exp(np.log(S0) + np.cumsum((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z, axis=1))
    cash = np.maximum(K - S[:, -1], 0)
    for t in range(steps - 2, -1, -1):          # backward induction = DP
        cash *= np.exp(-r * dt)
        itm = K - S[:, t] > 0
        if not itm.any(): continue
        x = S[itm, t] / K
        X = np.column_stack([np.ones_like(x), x, x**2])        # basis functions
        beta, *_ = np.linalg.lstsq(X, cash[itm], rcond=None)   # E[continuation | S_t]
        exercise = K - S[itm, t]
        ex_now = exercise > X @ beta
        cash[np.where(itm)[0][ex_now]] = exercise[ex_now]
    return np.exp(-r * dt) * cash.mean()
```

Verified on the LS (2001) test case S0=36, K=40, r=6%, σ=20%, T=1. The
converged binomial lattice gives **4.4867** (identical at 2 000 and 4 000
steps). LSM with 50 exercise dates and 100k paths gives **4.4678**. LSM is
**biased low** for two reasons: it prices a Bermudan option with 50
exercise dates rather than continuous exercise, and its regressed exercise
policy is suboptimal. The standard fix is to estimate the policy on one
set of paths and price on an independent set, which gives a clean lower
bound. For an upper bound, use the Andersen–Broadie duality method.

## 7. DP versus related paradigms

| Paradigm | Subproblems | Choice |
|---|---|---|
| Divide & conquer | Independent (no overlap) | — |
| **Dynamic programming** | Overlapping | Try all choices, keep the best |
| Greedy | One subproblem per step | Commit to a locally best choice. Correct only with the greedy-choice property (e.g. coin change works for US coins but fails for {1, 3, 4} with target 6) |
| Backtracking / branch & bound | Search tree | Prune infeasible or dominated branches |

## 8. Common pitfalls

- **Wrong or incomplete state.** If the answer depends on something not in
  the state, the recurrence is wrong, even if it passes small tests.
- **Wrong loop direction** in knapsack (0/1 vs unbounded).
- **Off-by-one errors** when mapping `dp[i]` (prefix length) to `a[i-1]`
  (element index).
- **Mutable defaults or unhashable arguments** with `lru_cache`. Pass
  tuples, not lists.
- **Recursion depth** in top-down Python. Use `sys.setrecursionlimit`
  cautiously, or convert to bottom-up.
- **Pseudo-polynomial is not polynomial.** O(n·W) knapsack is exponential
  in the *bit length* of W, and knapsack remains NP-hard.
- **Forgetting reconstruction.** If you need the actual path, keep
  back-pointers or re-derive the choices from the table.

## 9. When to reach for DP (quick checklist)

- The problem asks for an optimal value (min/max), a count of ways, or
  feasibility (yes/no).
- A brute-force recursion re-solves the same arguments repeatedly.
- The choices at each step depend only on a compact summary of the past
  (the Markov property). That summary is your state.
- The state space is small enough to enumerate. If it isn't, consider
  approximate DP, RL or LSM (§6), or a different paradigm.

## Sources

- [Why Is It Called "Dynamic Programming"? (Conversable Economist)](https://conversableeconomist.com/2022/08/24/why-is-it-called-dynamic-programming/)
- [Richard Bellman on the Birth of Dynamic Programming (Dreyfus, 2002)](https://www.researchgate.net/publication/220243993_Richard_Bellman_on_the_Birth_of_Dynamic_Programming)
- [The Principle of Optimality in Dynamic Programming: A Pedagogical Note (arXiv 2302.08467)](https://arxiv.org/pdf/2302.08467)
- [Longstaff & Schwartz (2001), Valuing American Options by Simulation: A Simple Least-Squares Approach](http://galton.uchicago.edu/~mykland/346W07/Longstaff.pdf)
- [Mike Giles, Advanced Monte Carlo Methods: American Options (Oxford lecture notes)](https://people.maths.ox.ac.uk/~gilesm/mc/module_6/american.pdf)
- Background: Cormen, Leiserson, Rivest & Stein, *Introduction to Algorithms*, ch. 14 (4th ed.) "Dynamic Programming"; Sutton & Barto, *Reinforcement Learning: An Introduction*, ch. 4 "Dynamic Programming"; Bertsekas, *Dynamic Programming and Optimal Control*.
