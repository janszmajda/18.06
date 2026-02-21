# 18.06 Spring 2026
# Problem Set 3
# Name: Jan Szmajda
# Collaborators: Codex 5.3

import numpy as np
from scipy.linalg import null_space, qr


def ordered_matrix(n: int, fill_by: str = "row") -> np.ndarray:
    if n < 1:
        raise ValueError("n must be >= 1")

    values = np.arange(1, n * n + 1, dtype=float)
    if fill_by == "row":
        return values.reshape((n, n))
    if fill_by == "column":
        return values.reshape((n, n), order="F")
    raise ValueError("fill_by must be 'row' or 'column'")


def observed_rank(n: int, fill_by: str = "row") -> int:
    A = ordered_matrix(n, fill_by=fill_by)
    return int(np.linalg.matrix_rank(A))


def theoretical_rank(n: int) -> int:
    return 1 if n == 1 else 2


def run_experiment(n_values: list[int], fill_by: str = "row") -> list[tuple[int, int, int, bool]]:
    rows: list[tuple[int, int, int, bool]] = []
    for n in n_values:
        obs = observed_rank(n, fill_by=fill_by)
        theo = theoretical_rank(n)
        rows.append((n, obs, theo, obs == theo))
    return rows


def graph_example_q2() -> tuple[int, list[tuple[int, int]]]:
    m = 5
    edges = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 1),
        (2, 5),
        (3, 1),
    ]
    return m, edges


def is_connected_undirected(m: int, edges: list[tuple[int, int]]) -> bool:
    adjacency: list[set[int]] = [set() for _ in range(m)]
    for u, v in edges:
        uu = u - 1
        vv = v - 1
        adjacency[uu].add(vv)
        adjacency[vv].add(uu)

    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == m


def is_strongly_connected(m: int, edges: list[tuple[int, int]]) -> bool:
    adjacency: list[list[int]] = [[] for _ in range(m)]
    for u, v in edges:
        adjacency[u - 1].append(v - 1)

    for start in range(m):
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if len(seen) != m:
            return False
    return True


def incidence_matrix(m: int, edges: list[tuple[int, int]]) -> np.ndarray:
    A = np.zeros((m, len(edges)), dtype=int)
    seen: set[tuple[int, int]] = set()

    for j, (u, v) in enumerate(edges):
        if not (1 <= u <= m and 1 <= v <= m):
            raise ValueError("Edge endpoints must be valid vertex labels in 1..m.")
        if u == v:
            raise ValueError("No self-loops allowed in this simple graph example.")
        if (u, v) in seen:
            raise ValueError("Duplicate directed edge found.")
        seen.add((u, v))

        A[u - 1, j] = -1
        A[v - 1, j] = 1

    return A


def cycle_basis_example() -> np.ndarray:
    z1 = np.array([1, 1, 1, 1, 1, 0, 0], dtype=int)
    z2 = np.array([1, 0, 0, 0, 1, 1, 0], dtype=int)
    z3 = np.array([1, 1, 0, 0, 0, 0, 1], dtype=int)
    return np.column_stack((z1, z2, z3))


def print_question_2() -> None:
    m, edges = graph_example_q2()
    n = len(edges)
    connected_undirected = is_connected_undirected(m, edges)
    connected_directed = is_strongly_connected(m, edges)
    A = incidence_matrix(m, edges)
    A_float = A.astype(float)

    rank_A = int(np.linalg.matrix_rank(A_float))
    N_numeric = null_space(A_float)
    nullity = N_numeric.shape[1]

    _, _, piv = qr(A_float, mode="economic", pivoting=True)
    pivot_cols = piv[:rank_A]
    col_basis = A[:, pivot_cols]

    C = cycle_basis_example()
    C_rank = int(np.linalg.matrix_rank(C.astype(float)))
    AZ = A @ C

    print("\n" + "=" * 72)
    print("QUESTION 2: DIRECTED GRAPH + INCIDENCE MATRIX")
    print("=" * 72)

    print("(a)")
    print(f"m = {m}")
    print(f"n = {n}")
    print(f"connected_directed = {connected_directed}")
    print(f"connected_undirected = {connected_undirected}")
    print("edges:")
    for idx, (u, v) in enumerate(edges, start=1):
        print(f"e{idx}: {u} -> {v}")
    print("digraph G { 1->2; 2->3; 3->4; 4->5; 5->1; 2->5; 3->1; }")

    print("\n(b)")
    print("A =")
    print(A)
    print("column_order = [e1, e2, e3, e4, e5, e6, e7]")

    print("\n(c)")
    print(f"rank_A = {rank_A}")
    print(f"nullity_A = {nullity}")
    print("nullspace_basis_numeric =")
    print(np.round(N_numeric, 6))
    print(f"norm_A_times_numeric_basis = {np.linalg.norm(A_float @ N_numeric):.2e}")
    print("nullspace_basis_integer =")
    print(C)
    print(f"rank_integer_basis = {C_rank}")
    print("A_times_integer_basis =")
    print(AZ)

    print("\n(d)")
    print(f"dim_col_A = {rank_A}")
    print(f"col_A_equals_Rm = {rank_A == m}")
    print(f"pivot_columns_1_indexed = {[int(i) + 1 for i in pivot_cols]}")
    print("col_space_basis =")
    print(col_basis)
    ones = np.ones(m, dtype=int)
    print("ones_T_A =")
    print(ones @ A)
    print("constraint_for_b: sum(b) = 0")
    if connected_undirected and rank_A == m - 1:
        print(f"col_A = {{b in R^{m} : sum(b) = 0}}")

    print("\n(e)")
    b = A[:, 0].astype(float)
    x_particular = np.zeros(n, dtype=float)
    x_particular[0] = 1.0
    print("b =")
    print(b.astype(int))
    print("x_particular =")
    print(x_particular.astype(int))
    print("general_solution:")
    print("x = x_particular + s*z1 + t*z2 + u*z3")
    sample_params = np.array([2.0, -1.0, 0.5])
    x_sample = x_particular + C.astype(float) @ sample_params
    residual = A_float @ x_sample - b
    print("sample_params = (2, -1, 0.5)")
    print(f"norm_Ax_minus_b = {np.linalg.norm(residual):.2e}")


def main() -> None:
    np.set_printoptions(suppress=True)

    n_values = list(range(1, 13))
    rows = run_experiment(n_values=n_values, fill_by="row")

    print("=" * 72)
    print("QUESTION 1: RANK OF n x n MATRIX WITH ENTRIES 1,2,...,n^2")
    print("=" * 72)
    print("fill = row-major\n")
    example_n = 4
    example_A = ordered_matrix(example_n, fill_by="row").astype(int)
    print(f"Example matrix for n = {example_n}:")
    print(example_A)
    print()

    print(" n   observed_rank   theoretical_rank   match")
    for n, obs, theo, ok in rows:
        print(f"{n:2d}        {obs:2d}                {theo:2d}         {ok}")

    print_question_2()


if __name__ == "__main__":
    main()
