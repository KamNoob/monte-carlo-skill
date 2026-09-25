"""Two-sequence DP: dp[i][j] over prefixes of two sequences a[:i], b[:j].

O(m*n) time and space in the basic form. Space can be cut to O(min(m,n))
if you only need the final number (see `edit_distance_linear_space`); if
you also need the actual alignment/edits, Hirschberg's algorithm gets
reconstruction down to O(m+n) space at the cost of two passes instead of
one (not implemented here — see references/optimization-techniques.md).
"""
from __future__ import annotations


def lcs_length(a: str, b: str) -> int:
    """Longest Common Subsequence length (not necessarily contiguous).

    dp[i][j] = dp[i-1][j-1] + 1            if a[i-1] == b[j-1]
             = max(dp[i-1][j], dp[i][j-1])  otherwise
    """
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def lcs_string(a: str, b: str) -> str:
    """Reconstructs one actual LCS (there can be several of the same length)."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    i, j = m, n
    out = []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1])
            i, j = i - 1, j - 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(out))


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: min single-char insert/delete/substitute edits.

    dp[i][j] = dp[i-1][j-1]                        if a[i-1] == b[j-1]  (match, free)
             = 1 + min(dp[i-1][j],                  delete from a
                        dp[i][j-1],                  insert into a
                        dp[i-1][j-1])                substitute
    Base cases dp[i][0]=i, dp[0][j]=j: turning a length-k prefix into/from
    the empty string costs k deletions/insertions.
    """
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def edit_distance_linear_space(a: str, b: str) -> int:
    """Same result as edit_distance, O(min(m,n)) space via rolling rows.

    Only the previous row is ever read, so keep two 1-D rows instead of
    the full 2-D table. Cannot reconstruct the alignment from this alone.
    """
    if len(a) < len(b):
        a, b = b, a  # make b the shorter dimension -> smaller row
    n = len(b)
    prev = list(range(n + 1))
    for i in range(1, len(a) + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n]


if __name__ == "__main__":
    assert lcs_length("ABCBDAB", "BDCABA") == 4  # e.g. BCBA
    assert lcs_string("ABCBDAB", "BDCABA") in ("BCBA", "BDAB", "BCAB")
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0
    for x, y in [("kitten", "sitting"), ("", "abc"), ("intention", "execution")]:
        assert edit_distance(x, y) == edit_distance_linear_space(x, y)
    print("sequence_alignment: all checks passed")
