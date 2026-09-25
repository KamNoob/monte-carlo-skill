# Two-Sequence Alignment DP

Reference script: `../../dynamic-programming-workspace/reference-scripts/sequence_alignment.py`.

State: `dp[i][j]`, defined over prefixes `a[:i]` and `b[:j]` of two sequences. O(m·n) time and space in the basic form.

## Longest Common Subsequence (LCS)

Longest subsequence (not necessarily contiguous) appearing in both `a` and `b`:

```
dp[i][j] = dp[i-1][j-1] + 1              if a[i-1] == b[j-1]
         = max(dp[i-1][j], dp[i][j-1])   otherwise
```

If the last characters match, they're worth including and the problem shrinks on both sides at once. If they don't match, the best LCS of the two prefixes is the better of "drop the last character of `a`" or "drop the last character of `b`" — at least one of them can safely be dropped without losing the true optimum, which is exactly the optimal-substructure argument that justifies the recurrence.

To reconstruct the actual subsequence (not just its length), walk backward from `dp[m][n]`: on a match, take the character and move diagonally; otherwise move toward whichever neighbor (`dp[i-1][j]` or `dp[i][j-1]`) is larger. There can be multiple LCSs of the same length — this reconstruction returns one of them, not necessarily a canonical choice.

## Edit distance (Levenshtein)

Minimum number of single-character insertions, deletions, and substitutions to turn `a` into `b`:

```
dp[i][j] = dp[i-1][j-1]                                  if a[i-1] == b[j-1]   (free match)
         = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])  otherwise            (delete / insert / substitute)
```

Base cases: `dp[i][0] = i`, `dp[0][j] = j` — turning a length-`k` prefix into/from the empty string costs exactly `k` insertions or deletions.

This is the DP behind `diff`-style tools and, with a substitution-cost/scoring-matrix generalization, **Needleman-Wunsch** (global alignment) and **Smith-Waterman** (local alignment) in bioinformatics — same recurrence shape, different cost function and (for Smith-Waterman) a floor of 0 on `dp[i][j]` so alignment can restart anywhere.

## Space optimization: rolling rows

Both LCS and edit distance only ever read the *previous* row (and the current row's already-computed left neighbor) to fill in the current row — so the full `m × n` table can be replaced by two 1-D rows, cutting space from O(m·n) to O(min(m,n)) (make the shorter sequence the inner dimension). `sequence_alignment.py`'s `edit_distance_linear_space` does this and is cross-checked against the full table.

**The catch:** rolling rows gives you the final *number* but destroys the information needed to reconstruct the actual alignment (the traceback needs the whole table, or at least the diagonal history). If you need both linear space and the actual alignment, use **Hirschberg's algorithm** (divide and conquer: find the midpoint row of the alignment using two linear-space forward/backward passes, recurse on each half) — O(m+n) space, same O(mn) time, at the cost of a small constant-factor slowdown from the two passes. See `references/optimization-techniques.md`.

## When to reach for this family

Two sequences being compared, aligned, or diffed, where the answer depends on matching or mismatching pairs of elements across both prefixes. If there's only one sequence and the question is about a subsequence or partition of it alone (not against another sequence), you likely want 1-D DP or interval DP instead — see `references/linear-and-knapsack-dp.md` and `references/interval-dp.md`.
