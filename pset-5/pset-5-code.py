# 18.06 Spring 2026
# Problem Set 5
# Name: Jan Szmajda
# Collaborators: Codex 5.3

from __future__ import annotations

import numpy as np


def givens_coefficients(a: float, b: float) -> tuple[float, float, float]:
    """
    Return c, s, r so that

        [[c, s], [-s, c]] @ [a, b]^T = [r, 0]^T

    with c^2 + s^2 = 1 and r >= 0 whenever (a, b) != (0, 0).
    """
    r = float(np.hypot(a, b))
    if r == 0.0:
        return 1.0, 0.0, 0.0
    return float(a / r), float(b / r), r


def givens_matrix(m: int, i: int, j: int, c: float, s: float) -> np.ndarray:
    G = np.eye(m, dtype=float)
    G[i, i] = c
    G[i, j] = s
    G[j, i] = -s
    G[j, j] = c
    return G


def apply_left_givens(
    A: np.ndarray,
    i: int,
    j: int,
    c: float,
    s: float,
    start_col: int = 0,
) -> None:
    rows = A[[i, j], start_col:].copy()
    A[i, start_col:] = c * rows[0] + s * rows[1]
    A[j, start_col:] = -s * rows[0] + c * rows[1]


def givens_qr(A: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, float, float]]]:
    A = np.array(A, dtype=float)
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m, dtype=float)
    rotations: list[tuple[int, int, float, float]] = []

    for col in range(min(m, n)):
        for row in range(m - 1, col, -1):
            if abs(R[row, col]) <= tol:
                continue

            c, s, _ = givens_coefficients(R[col, col], R[row, col])
            apply_left_givens(R, col, row, c, s, start_col=col)

            # Since R = G_k ... G_1 A, accumulate Q = G_1^T ... G_k^T.
            cols = Q[:, [col, row]].copy()
            Q[:, col] = c * cols[:, 0] + s * cols[:, 1]
            Q[:, row] = -s * cols[:, 0] + c * cols[:, 1]

            R[row, col] = 0.0
            rotations.append((col, row, c, s))

    R[np.abs(R) <= tol] = 0.0
    Q[np.abs(Q) <= tol] = 0.0
    return Q, R, rotations


def max_below_diagonal(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.tril(A, k=-1))))


def random_orthogonal_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(Z)
    return Q


def householder_vector(x: np.ndarray, alpha: float | None = None) -> tuple[np.ndarray, float]:
    x = np.array(x, dtype=float)
    norm_x = float(np.linalg.norm(x))
    if norm_x == 0.0:
        raise ValueError("Householder vector is undefined for the zero vector.")

    if alpha is None:
        lead = x[0] if x[0] != 0.0 else 1.0
        alpha = float(-np.copysign(norm_x, lead))

    v = x.copy()
    v[0] -= alpha
    v_norm = float(np.linalg.norm(v))
    if v_norm == 0.0:
        raise ValueError("Degenerate Householder construction.")

    return v / v_norm, float(alpha)


def householder_matrix(u: np.ndarray) -> np.ndarray:
    return np.eye(u.size, dtype=float) - 2.0 * np.outer(u, u)


def embedded_householder(m: int, start: int, u: np.ndarray) -> np.ndarray:
    H = np.eye(m, dtype=float)
    H[start:, start:] -= 2.0 * np.outer(u, u)
    return H


def apply_left_householder(
    A: np.ndarray,
    u: np.ndarray,
    start_row: int,
    start_col: int = 0,
) -> None:
    sub = A[start_row:, start_col:].copy()
    A[start_row:, start_col:] = sub - 2.0 * np.outer(u, u @ sub)


def apply_right_householder(
    A: np.ndarray,
    u: np.ndarray,
    start_col: int,
) -> None:
    sub = A[:, start_col:].copy()
    A[:, start_col:] = sub - 2.0 * np.outer(sub @ u, u)


def householder_qr(A: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, list[tuple[int, np.ndarray, float]]]:
    A = np.array(A, dtype=float)
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m, dtype=float)
    reflectors: list[tuple[int, np.ndarray, float]] = []

    for col in range(min(m - 1, n)):
        x = R[col:, col]
        if np.linalg.norm(x[1:]) <= tol:
            continue

        u, alpha = householder_vector(x)
        apply_left_householder(R, u, col, col)
        apply_right_householder(Q, u, col)

        R[col + 1:, col] = 0.0
        reflectors.append((col, u.copy(), alpha))

    R[np.abs(R) <= tol] = 0.0
    Q[np.abs(Q) <= tol] = 0.0
    return Q, R, reflectors


def givens_part_a_data() -> tuple[float, float, float, float]:
    v = np.array([3.0, 4.0], dtype=float)
    c, s, r = givens_coefficients(v[0], v[1])
    theta = float(np.arctan2(v[1], v[0]))
    return c, s, r, theta


def givens_part_b_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A = np.array(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, -1.0],
        ],
        dtype=float,
    )

    c21, s21, _ = givens_coefficients(A[0, 0], A[1, 0])
    G21 = givens_matrix(3, 0, 1, c21, s21)
    A1 = G21 @ A

    c31, s31, _ = givens_coefficients(A1[0, 0], A1[2, 0])
    G31 = givens_matrix(3, 0, 2, c31, s31)
    A2 = G31 @ A1

    c32, s32, _ = givens_coefficients(A2[1, 1], A2[2, 1])
    G32 = givens_matrix(3, 1, 2, c32, s32)
    R = G32 @ A2

    R[np.abs(R) <= 1e-12] = 0.0
    return A, G21, G31, G32, R


def qr_test_matrices() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(1806)
    return [
        (
            "part_b_matrix",
            np.array(
                [
                    [1.0, 1.0],
                    [1.0, 0.0],
                    [1.0, -1.0],
                ],
                dtype=float,
            ),
        ),
        (
            "square_3x3",
            np.array(
                [
                    [2.0, -1.0, 3.0],
                    [1.0, 0.0, 4.0],
                    [5.0, 2.0, 1.0],
                ],
                dtype=float,
            ),
        ),
        (
            "tall_4x3",
            np.array(
                [
                    [1.0, 2.0, 0.0],
                    [-1.0, 3.0, 4.0],
                    [2.0, 0.0, 1.0],
                    [1.0, -2.0, 5.0],
                ],
                dtype=float,
            ),
        ),
        (
            "wide_3x4",
            np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [0.0, -1.0, 2.0, 1.0],
                    [5.0, 1.0, 0.0, -2.0],
                ],
                dtype=float,
            ),
        ),
        ("random_5x4", rng.normal(size=(5, 4))),
    ]


def orthogonal_test_cases() -> list[tuple[str, np.ndarray]]:
    return [
        ("random_orthogonal_n5", random_orthogonal_matrix(5, seed=2026)),
        ("sign_matrix_n5", np.diag([-1.0, 1.0, -1.0, 1.0, 1.0])),
    ]


def householder_part_a_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = np.array([3.0, 4.0], dtype=float)
    u, _ = householder_vector(v, alpha=-float(np.linalg.norm(v)))
    Q = householder_matrix(u)
    u_alt, _ = householder_vector(v, alpha=float(np.linalg.norm(v)))
    return v, u, Q @ v, u_alt


def householder_part_b_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A = np.array(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, -1.0],
        ],
        dtype=float,
    )

    u2, _ = householder_vector(A[:, 0])
    H2 = embedded_householder(3, 0, u2)
    A1 = H2 @ A

    u1, _ = householder_vector(A1[1:, 1])
    H1 = embedded_householder(3, 1, u1)
    R = H1 @ A1

    R[np.abs(R) <= 1e-12] = 0.0
    H1[np.abs(H1) <= 1e-12] = 0.0
    H2[np.abs(H2) <= 1e-12] = 0.0
    return A, H1, H2, R


def main() -> None:
    np.set_printoptions(precision=8, suppress=True)

    c, s, r, theta = givens_part_a_data()

    print("=" * 72)
    print("QUESTION 1(a): SINGLE GIVENS ROTATION IN R^2")
    print("=" * 72)
    print("v = [3, 4]^T")
    print(f"c = {c:.8f} = 3/5")
    print(f"s = {s:.8f} = 4/5")
    print(f"r = {r:.8f} = 5")
    print("Check:")
    print(np.array([[c, s], [-s, c]]) @ np.array([3.0, 4.0]))
    print(
        "\nGeometric interpretation: c = cos(theta), s = sin(theta), "
        f"where theta = atan2(4, 3) = {theta:.8f} radians."
    )
    print("The matrix [[c, s], [-s, c]] is a clockwise rotation by theta, sending v to [5, 0]^T.")

    A, G21, G31, G32, R_b = givens_part_b_data()
    print("\n" + "=" * 72)
    print("QUESTION 1(b): EXPLICIT ROTATIONS FOR THE GIVEN 3x2 MATRIX")
    print("=" * 72)
    print("A =")
    print(A)
    print("\nG21 =")
    print(G21)
    print("\nG31 =")
    print(G31)
    print("\nG32 =")
    print(G32)
    print("\nG32 G31 G21 A =")
    print(R_b)

    print("\n" + "=" * 72)
    print("QUESTION 1(c): QR DECOMPOSITION USING GIVENS ROTATIONS")
    print("=" * 72)
    for name, test_A in qr_test_matrices():
        Q, R, rotations = givens_qr(test_A)
        reconstruction_error = float(np.linalg.norm(Q @ R - test_A))
        orthogonality_error = float(np.linalg.norm(Q.T @ Q - np.eye(Q.shape[0])))
        lower_error = max_below_diagonal(R)

        print(f"\nTest matrix: {name}")
        print(f"shape(A) = {test_A.shape}")
        print(f"rotations used = {len(rotations)}")
        print(f"||Q R - A||_F = {reconstruction_error:.6e}")
        print(f"||Q^T Q - I||_F = {orthogonality_error:.6e}")
        print(f"max |entry below diagonal in R| = {lower_error:.6e}")

    print("\n" + "=" * 72)
    print("QUESTION 1(d): STARTING FROM AN ORTHOGONAL MATRIX")
    print("=" * 72)
    for name, orth_A in orthogonal_test_cases():
        Q, R, rotations = givens_qr(orth_A)
        diag_R = np.diag(R)

        print(f"\nExample: {name}")
        print(f"n = {orth_A.shape[0]}")
        print(f"rotations used = {len(rotations)}")
        print(f"generic dense upper bound n(n-1)/2 = {orth_A.shape[0] * (orth_A.shape[0] - 1) // 2}")
        print(f"||Q R - A||_F = {np.linalg.norm(Q @ R - orth_A):.6e}")
        print(f"||R^T R - I||_F = {np.linalg.norm(R.T @ R - np.eye(R.shape[0])):.6e}")
        print(f"max |entry below diagonal in R| = {max_below_diagonal(R):.6e}")
        print("diag(R) =")
        print(diag_R)

    print("\nConclusions for part (d):")
    print("- Each Givens rotation removes one subdiagonal entry, so a dense n x n matrix uses n(n-1)/2 rotations.")
    print("- If A is orthogonal, then R is both upper triangular and orthogonal, so R must be diagonal with diagonal entries +/-1.")
    print("- If one also enforces the usual positive-diagonal convention, then R = I and Q = A.")
    print("- Therefore any orthogonal matrix can be written as a product of Givens rotations, up to a diagonal sign matrix.")
    print("- In particular, any orthogonal matrix with determinant +1 is a product of Givens rotations alone.")

    v, u, qv, u_alt = householder_part_a_data()
    print("\n" + "=" * 72)
    print("QUESTION 2(a): SINGLE HOUSEHOLDER REFLECTION IN R^2")
    print("=" * 72)
    print("v =")
    print(v)
    print("u =")
    print(u)
    print("Check Qv with Q = I - 2uu^T:")
    print(qv)
    print("An alternate valid choice sends v to [5, 0]^T, with u_alt =")
    print(u_alt)

    A_h, H1, H2, R_h = householder_part_b_data()
    print("\n" + "=" * 72)
    print("QUESTION 2(b): EXPLICIT HOUSEHOLDER REFLECTIONS")
    print("=" * 72)
    print("A =")
    print(A_h)
    print("\nH2 =")
    print(H2)
    print("\nH1 =")
    print(H1)
    print("\nH1 H2 A =")
    print(R_h)

    print("\n" + "=" * 72)
    print("QUESTION 2(c): QR DECOMPOSITION USING HOUSEHOLDER REFLECTIONS")
    print("=" * 72)
    for name, test_A in qr_test_matrices():
        Q, R, reflectors = householder_qr(test_A)
        reconstruction_error = float(np.linalg.norm(Q @ R - test_A))
        orthogonality_error = float(np.linalg.norm(Q.T @ Q - np.eye(Q.shape[0])))
        lower_error = max_below_diagonal(R)

        print(f"\nTest matrix: {name}")
        print(f"shape(A) = {test_A.shape}")
        print(f"reflectors used = {len(reflectors)}")
        print(f"||Q R - A||_F = {reconstruction_error:.6e}")
        print(f"||Q^T Q - I||_F = {orthogonality_error:.6e}")
        print(f"max |entry below diagonal in R| = {lower_error:.6e}")

    print("\nNumerical-stability note:")
    print("- For a column x, choose alpha = -sign(x_1) ||x|| so x_1 - alpha is large in magnitude.")
    print("- This avoids subtracting nearly equal numbers when forming the reflector vector.")
    print("- In part (a), targeting [-5, 0]^T uses v - (-5)e1 = [8, 4]^T, while targeting [5, 0]^T uses v - 5e1 = [-2, 4]^T.")

    print("\n" + "=" * 72)
    print("QUESTION 2(d): STARTING FROM AN ORTHOGONAL MATRIX")
    print("=" * 72)
    for name, orth_A in orthogonal_test_cases():
        Q, R, reflectors = householder_qr(orth_A)
        diag_R = np.diag(R)

        print(f"\nExample: {name}")
        print(f"n = {orth_A.shape[0]}")
        print(f"reflectors used = {len(reflectors)}")
        print(f"generic dense upper bound n-1 = {orth_A.shape[0] - 1}")
        print(f"||Q R - A||_F = {np.linalg.norm(Q @ R - orth_A):.6e}")
        print(f"||R^T R - I||_F = {np.linalg.norm(R.T @ R - np.eye(R.shape[0])):.6e}")
        print(f"max |entry below diagonal in R| = {max_below_diagonal(R):.6e}")
        print("diag(R) =")
        print(diag_R)

    print("\nConclusions for Question 2(d):")
    print("- Householder QR uses at most n-1 reflections for an n x n matrix, one per column except the last.")
    print("- If A is orthogonal, then R is upper triangular and orthogonal, so R must be diagonal with entries +/-1.")
    print("- Hence A = Q R with Q a product of Householder reflections and R a diagonal sign matrix.")
    print("- Since each coordinate sign flip is itself a Householder reflection, every orthogonal matrix is a product of Householder reflections.")
    print("- In fact, at most n Householder reflections are needed, and the parity must match det(A).")


if __name__ == "__main__":
    main()
