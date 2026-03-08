# 18.06 Spring 2026
# Problem Set 4
# Name: Jan Szmajda
# Collaborators: Codex 5.3

from __future__ import annotations

import numpy as np


def square_graph_adjacency() -> np.ndarray:
    # 4-cycle: 1-2-3-4-1
    return np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ],
        dtype=np.int8,
    )


def random_undirected_graph_adjacency(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be >= 1")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must satisfy 0 <= p <= 1")

    upper = np.triu(rng.random((n, n)) < p, k=1)
    A = upper.astype(np.int8)
    A = A + A.T
    return A


def connected_and_diameter_via_powers(A: np.ndarray) -> tuple[bool, int | None, np.ndarray]:
    """
    Use adjacency-matrix powers in the boolean sense:
    W_k[i,j] = 1 iff there exists a walk of length k from i to j.
    Distances are the first k where W_k[i,j] becomes 1.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    n = A.shape[0]
    A01 = (A > 0).astype(np.int8)

    reachable = np.eye(n, dtype=bool)        # I + A + ... + A^k > 0 (boolean)
    walks_k = np.eye(n, dtype=np.int8)       # W_0 = I
    dist = np.zeros((n, n), dtype=int)       # shortest distances (0 on diagonal)

    for k in range(1, n):
        # Boolean product via standard multiplication + threshold.
        walks_k = ((walks_k @ A01) > 0).astype(np.int8)
        walks_bool = walks_k.astype(bool)

        newly_reached = walks_bool & (~reachable)
        dist[newly_reached] = k
        reachable |= walks_bool

        if reachable.all():
            return True, int(dist.max()), dist

    return False, None, dist


def estimate_for_np(
    n: int,
    p: float,
    trials: int,
    seed: int,
) -> tuple[float, dict[int, float], int]:
    rng = np.random.default_rng(seed)
    connected_count = 0
    diam_counts: dict[int, int] = {}

    for _ in range(trials):
        A = random_undirected_graph_adjacency(n=n, p=p, rng=rng)
        connected, diameter, _ = connected_and_diameter_via_powers(A)
        if connected:
            connected_count += 1
            assert diameter is not None
            diam_counts[diameter] = diam_counts.get(diameter, 0) + 1

    prob_connected = connected_count / trials
    if connected_count == 0:
        return prob_connected, {}, connected_count

    diameter_distribution = {
        d: count / connected_count for d, count in sorted(diam_counts.items())
    }
    return prob_connected, diameter_distribution, connected_count


def print_part_a_b_c() -> None:
    A = square_graph_adjacency()

    print("=" * 72)
    print("COMPUTE 1: ADJACENCY MATRIX, CONNECTEDNESS, DIAMETER")
    print("=" * 72)

    print("(a)")
    print("Square graph (cycle C4) with vertices {1,2,3,4} and 4 edges:")
    print("(1,2), (2,3), (3,4), (4,1)")
    print("Adjacency matrix A =")
    print(A)

    A2 = A @ A
    A3 = A2 @ A
    print("\n(b)")
    print("A^2 =")
    print(A2)
    print("Interpretation: (A^2)_{ij} is the number of length-2 walks from i to j.")
    print("More generally, (A^n)_{ij} is the number of length-n walks from i to j.")
    print("A^3 =")
    print(A3)

    connected, diameter, dist = connected_and_diameter_via_powers(A)
    print("\n(c)")
    print("Method using powers:")
    print("B = I + A + A^2 + ... + A^(n-1). Graph is connected iff every entry of B is > 0.")
    print("Distances come from the first k where (A^k)_{ij} > 0; diameter is max distance.")
    print(f"connected = {connected}")
    print(f"diameter = {diameter}")
    print("distance_matrix =")
    print(dist)


def print_part_d() -> None:
    print("\n(d)")
    print("Random graph model G(n,p): add each edge independently with probability p.")

    trials = 300
    cases = [
        (20, 0.05),
        (20, 0.10),
        (20, 0.20),
        (40, 0.05),
        (40, 0.10),
        (40, 0.20),
    ]

    print(f"trials_per_case = {trials}")
    for idx, (n, p) in enumerate(cases):
        seed = 2026 + idx
        prob_connected, diam_dist, connected_count = estimate_for_np(
            n=n,
            p=p,
            trials=trials,
            seed=seed,
        )

        print(f"\nCase n={n}, p={p:.2f}:")
        print(f"P(connected) ≈ {prob_connected:.4f} ({connected_count}/{trials})")
        if connected_count == 0:
            print("No connected samples; diameter distribution unavailable.")
            continue

        print("Estimated diameter distribution conditioned on connected:")
        for d, prob in diam_dist.items():
            print(f"  P(diameter={d} | connected) ≈ {prob:.4f}")


def random_binary_matrix(
    m: int,
    n: int,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if m < 1 or n < 1:
        raise ValueError("m and n must be >= 1")
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must satisfy 0 <= p <= 1")
    return (rng.random((m, n)) < p).astype(np.int8)


def rank_distribution(
    m: int,
    n: int,
    p: float,
    trials: int,
    seed: int,
) -> tuple[dict[int, float], float]:
    rng = np.random.default_rng(seed)
    counts: dict[int, int] = {}
    rank_sum = 0

    for _ in range(trials):
        A = random_binary_matrix(m=m, n=n, p=p, rng=rng)
        r = int(np.linalg.matrix_rank(A))
        counts[r] = counts.get(r, 0) + 1
        rank_sum += r

    distribution = {r: c / trials for r, c in sorted(counts.items())}
    expected_rank = rank_sum / trials
    return distribution, expected_rank


def expected_rank_vs_p(
    m: int,
    n: int,
    p_grid: np.ndarray,
    trials_per_p: int,
    base_seed: int,
) -> tuple[np.ndarray, float, float]:
    means: list[float] = []

    for j, p in enumerate(p_grid):
        _, mean_rank = rank_distribution(
            m=m,
            n=n,
            p=float(p),
            trials=trials_per_p,
            seed=base_seed + j,
        )
        means.append(mean_rank)

    means_arr = np.array(means, dtype=float)
    best_idx = int(np.argmax(means_arr))
    return means_arr, float(p_grid[best_idx]), float(means_arr[best_idx])


def print_compute_2() -> None:
    print("\n" + "=" * 72)
    print("COMPUTE 2: RANDOM 0-1 MATRICES AND FUNDAMENTAL SUBSPACES")
    print("=" * 72)
    print("If A is m x n with rank r, then:")
    print("dim(Col A) = r, dim(Row A) = r, dim(N(A)) = n-r, dim(N(A^T)) = m-r")

    print("\nPart 1: Distributions for several shapes and probabilities")
    trials_dist = 800
    dist_shapes = [(8, 8), (12, 8), (8, 12)]
    dist_ps = [0.10, 0.30, 0.50, 0.70, 0.90]

    print(f"trials_per_case = {trials_dist}")
    for i, (m, n) in enumerate(dist_shapes):
        for j, p in enumerate(dist_ps):
            seed = 5000 + 100 * i + j
            dist, mean_rank = rank_distribution(m=m, n=n, p=p, trials=trials_dist, seed=seed)

            print(f"\nShape m={m}, n={n}, p={p:.2f}:")
            print(f"E[rank] ≈ {mean_rank:.4f}")
            print("Distribution of fundamental-subspace dimensions:")
            for r, prob in dist.items():
                dim_col = r
                dim_row = r
                dim_null = n - r
                dim_left_null = m - r
                print(
                    f"  P(r={r})≈{prob:.4f}  -> "
                    f"(dim Col={dim_col}, dim Row={dim_row}, "
                    f"dim Null={dim_null}, dim LeftNull={dim_left_null})"
                )

    print("\nPart 2: Estimate p that maximizes expected rank")
    trials_opt = 500
    p_grid = np.linspace(0.05, 0.95, 19)
    opt_shapes = [(8, 8), (12, 8), (8, 12), (16, 16)]
    print(f"trials_per_p = {trials_opt}")
    print(f"p_grid = {[round(float(p), 2) for p in p_grid]}")

    for k, (m, n) in enumerate(opt_shapes):
        means, best_p, best_mean = expected_rank_vs_p(
            m=m,
            n=n,
            p_grid=p_grid,
            trials_per_p=trials_opt,
            base_seed=8000 + 500 * k,
        )
        print(f"\nShape m={m}, n={n}:")
        print(f"Estimated p maximizing E[rank] is p* ≈ {best_p:.2f}")
        print(f"max E[rank] ≈ {best_mean:.4f}")

        top_idx = np.argsort(means)[-3:][::-1]
        print("Top 3 p values by estimated E[rank]:")
        for idx in top_idx:
            print(f"  p={p_grid[idx]:.2f} -> E[rank]≈{means[idx]:.4f}")


def main() -> None:
    np.set_printoptions(suppress=True)
    print_part_a_b_c()
    print_part_d()
    print_compute_2()


if __name__ == "__main__":
    main()
