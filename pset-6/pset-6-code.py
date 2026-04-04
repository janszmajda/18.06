# 18.06 Spring 2026
# Problem Set 6
# Name: Jan Szmajda
# Collaborators: Opus 4.6

from __future__ import annotations

import numpy as np
from itertools import permutations, product
from collections import Counter


def max_determinant_3x3() -> tuple[int, np.ndarray]:
    """
    Find the maximum determinant of a 3x3 matrix whose entries are
    a permutation of {1, 2, ..., 9}.
    Returns (max_det, best_matrix).
    """
    best_det = -np.inf
    best_matrix = None

    for perm in permutations(range(1, 10)):
        M = np.array(perm, dtype=int).reshape((3, 3))
        d = int(round(np.linalg.det(M.astype(float))))
        if d > best_det:
            best_det = d
            best_matrix = M

    return int(best_det), best_matrix


def print_compute_1() -> None:
    print("=" * 72)
    print("COMPUTE 1: MAXIMUM DETERMINANT OF 3x3 MATRIX WITH ENTRIES 1,...,9")
    print("=" * 72)

    max_det, best_matrix = max_determinant_3x3()

    print(f"Maximum determinant = {max_det}")
    print("Achieved by matrix:")
    print(best_matrix)
    print(f"\nVerification: det = {np.linalg.det(best_matrix.astype(float)):.6f}")


def enumerate_binary_det_distribution(n: int) -> dict[int, int]:
    """Enumerate all 2^(n^2) binary n×n matrices and count determinant values."""
    counts: dict[int, int] = Counter()
    for entries in product([0, 1], repeat=n * n):
        M = np.array(entries, dtype=int).reshape((n, n))
        d = int(round(np.linalg.det(M.astype(float))))
        counts[d] += 1
    return dict(sorted(counts.items()))


def sample_binary_det_distribution(
    n: int, trials: int, seed: int,
) -> tuple[dict[int, int], int, np.ndarray]:
    """Random-sample binary n×n matrices; return det counts, max det, and a maximizer."""
    rng = np.random.default_rng(seed)
    counts: dict[int, int] = Counter()
    best_det = 0
    best_matrix = np.zeros((n, n), dtype=int)

    for _ in range(trials):
        M = rng.integers(0, 2, size=(n, n))
        d = int(round(np.linalg.det(M.astype(float))))
        counts[d] += 1
        if abs(d) > abs(best_det):
            best_det = d
            best_matrix = M.copy()

    return dict(sorted(counts.items())), best_det, best_matrix


def print_compute_2() -> None:
    print("\n" + "=" * 72)
    print("COMPUTE 2: DETERMINANT DISTRIBUTION OF 0-1 MATRICES")
    print("=" * 72)

    # --- Exact enumeration for n = 1..4 ---
    for n in range(1, 5):
        total = 2 ** (n * n)
        dist = enumerate_binary_det_distribution(n)
        print(f"\nn = {n}  (total matrices = {total})")

        most_common_val = max(dist, key=lambda k: dist[k])
        max_det = max(dist.keys())
        min_det = min(dist.keys())

        print(f"Possible determinant values: {min_det} to {max_det}")
        print(f"Most common value: {most_common_val} (count = {dist[most_common_val]}, "
              f"fraction = {dist[most_common_val] / total:.4f})")
        print(f"Maximum |det| = {max(abs(min_det), abs(max_det))}")
        print("Full distribution:")
        for d, c in dist.items():
            print(f"  det={d:4d}: count={c:6d}  fraction={c / total:.6f}")

    # --- Random sampling for larger n ---
    trials = 100_000
    print(f"\nRandom sampling ({trials} trials per n):")
    for n in [5, 6, 7, 8]:
        dist, best_det, best_matrix = sample_binary_det_distribution(n, trials, seed=2026 + n)
        total_sampled = sum(dist.values())
        most_common_val = max(dist, key=lambda k: dist[k])
        max_abs = max(abs(d) for d in dist.keys())

        print(f"\nn = {n}:")
        print(f"Most common determinant value: {most_common_val} "
              f"(fraction = {dist[most_common_val] / total_sampled:.4f})")
        print(f"Largest |det| observed: {max_abs}")
        print(f"Best matrix found (det = {best_det}):")
        print(best_matrix)
        print(f"Number of distinct det values observed: {len(dist)}")


def main() -> None:
    np.set_printoptions(suppress=True)
    print_compute_1()
    print_compute_2()


if __name__ == "__main__":
    main()
