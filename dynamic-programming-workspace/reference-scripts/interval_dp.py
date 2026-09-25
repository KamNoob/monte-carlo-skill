"""Interval DP: dp[i][j] over a contiguous range/substring, split at some k.

Naive complexity is O(n^3): O(n^2) intervals, O(n) split points each.
Knuth's optimization can bring this to O(n^2) when the optimal split
point is monotone in i and j (quadrangle inequality) — not implemented
here, see references/optimization-techniques.md.
"""
from __future__ import annotations


def matrix_chain_order(dims: list[int]) -> tuple[int, list[list[int]]]:
    """Min scalar multiplications to fully parenthesize a chain of matrices.

    Matrix i has shape dims[i] x dims[i+1], for i in 0..n-1 (n matrices,
    n+1 dims). dp[i][j] = min cost to multiply matrices i..j (inclusive).

    dp[i][j] = min over split k in [i, j-1] of:
        dp[i][k] + dp[k+1][j] + dims[i]*dims[k+1]*dims[j+1]
    (cost of the left sub-chain, the right sub-chain, plus the cost of
    multiplying the two resulting matrices together).

    Returns (min_cost, split) where split[i][j] records the k achieving
    the minimum, for reconstructing the parenthesization.
    """
    n = len(dims) - 1  # number of matrices
    dp = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):  # length = number of matrices in the chain
        for i in range(0, n - length + 1):
            j = i + length - 1
            dp[i][j] = float("inf")
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k
    return dp[0][n - 1], split


def matrix_chain_parens(split: list[list[int]], i: int, j: int) -> str:
    """Reconstructs the optimal parenthesization as a string, e.g. '((A0 A1) A2)'."""
    if i == j:
        return f"A{i}"
    k = split[i][j]
    return f"({matrix_chain_parens(split, i, k)} {matrix_chain_parens(split, k + 1, j)})"


def min_cuts_palindrome_partition(s: str) -> int:
    """Fewest cuts to partition s so every piece is a palindrome.

    First precompute is_pal[i][j] (interval DP: a substring is a
    palindrome iff its ends match and the inside is a palindrome), then
    cuts[j] = min cuts for s[:j+1]:
        0                                    if s[:j+1] itself is a palindrome
        min(cuts[i-1] + 1 for i where s[i:j+1] is a palindrome)  otherwise
    """
    n = len(s)
    if n == 0:
        return 0
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and (length == 2 or is_pal[i + 1][j - 1]):
                is_pal[i][j] = True

    cuts = [0] * n
    for j in range(n):
        if is_pal[0][j]:
            cuts[j] = 0
            continue
        cuts[j] = j  # worst case: cut before every character
        for i in range(1, j + 1):
            if is_pal[i][j]:
                cuts[j] = min(cuts[j], cuts[i - 1] + 1)
    return cuts[n - 1]


if __name__ == "__main__":
    # Classic CLRS example: dims for A1(30x35) A2(35x15) A3(15x5) A4(5x10) A5(10x20) A6(20x25)
    dims = [30, 35, 15, 5, 10, 20, 25]
    cost, split = matrix_chain_order(dims)
    assert cost == 15125, cost
    parens = matrix_chain_parens(split, 0, len(dims) - 2)
    assert parens == "((A0 (A1 A2)) ((A3 A4) A5))", parens

    assert min_cuts_palindrome_partition("aab") == 1     # "aa" | "b"
    assert min_cuts_palindrome_partition("a") == 0
    assert min_cuts_palindrome_partition("racecar") == 0  # already a palindrome
    assert min_cuts_palindrome_partition("abcde") == 4    # every char its own piece
    print("interval_dp: all checks passed")
