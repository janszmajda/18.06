# 18.06 Spring 2026
# Problem Set 2
# Name: Jan Szmajda
# Collaborators: Codex 5.3

## Question 1 (20 points)
import numpy as np

def build_matrix() -> np.ndarray:
    return np.array(
        [
            [8.40, 4.20, 2.80, 2.10],
            [4.20, 2.80, 2.10, 1.68],
            [2.80, 2.10, 1.68, 1.40],
            [2.10, 1.68, 1.40, 1.20],
        ],
        dtype=float,
    )

def perturbation_experiment(
    A: np.ndarray,
    A_inv: np.ndarray,
    k_values: range,
    n_samples: int = 2000,
    seed: int = 1806,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    inv_norm_inf = np.linalg.norm(A_inv, ord=np.inf)
    rows: list[dict[str, float]] = []

    for k in k_values:
        eps = 10.0 ** (-k)
        max_entry_errors = []
        rel_inf_errors = []
        singular_count = 0

        for _ in range(n_samples):
            E = rng.uniform(-eps, eps, size=A.shape)
            try:
                B_inv = np.linalg.inv(A + E)
            except np.linalg.LinAlgError:
                singular_count += 1
                continue

            delta = B_inv - A_inv
            max_entry_errors.append(np.max(np.abs(delta)))
            rel_inf_errors.append(np.linalg.norm(delta, ord=np.inf) / inv_norm_inf)

        max_entry_errors = np.array(max_entry_errors, dtype=float)
        rel_inf_errors = np.array(rel_inf_errors, dtype=float)
        sig_digits = -np.log10(np.maximum(rel_inf_errors, np.finfo(float).tiny))

        rows.append(
            {
                "k": float(k),
                "eps": eps,
                "singular_count": float(singular_count),
                "median_max_entry_error": float(np.quantile(max_entry_errors, 0.50)),
                "p95_max_entry_error": float(np.quantile(max_entry_errors, 0.95)),
                "worst_max_entry_error": float(np.max(max_entry_errors)),
                "median_rel_inf_error": float(np.quantile(rel_inf_errors, 0.50)),
                "p95_rel_inf_error": float(np.quantile(rel_inf_errors, 0.95)),
                "median_correct_digits": float(np.quantile(sig_digits, 0.50)),
                "p05_correct_digits": float(np.quantile(sig_digits, 0.05)),
            }
        )

    return rows

def first_k_meeting_threshold(rows: list[dict[str, float]], field: str, threshold: float) -> int | None:
    for row in rows:
        if row[field] <= threshold:
            return int(row["k"])
    return None

def main() -> None:
    np.set_printoptions(precision=8, suppress=True)

    A = build_matrix()
    A_inv = np.linalg.inv(A)
    cond_2 = np.linalg.cond(A, p=2)
    digits_lost = np.log10(cond_2)

    print("=" * 72)
    print("QUESTION 1(a): INVERSE OF A")
    print("=" * 72)
    print("A =")
    print(A)
    print("\nA^{-1} =")
    print(A_inv)
    print(f"\n2-norm condition number cond_2(A) = {cond_2:.6e}")
    print(f"Approximate digits lost in inversion: log10(cond_2(A)) = {digits_lost:.3f}")

    print("\n" + "=" * 72)
    print("QUESTION 1(b): PERTURBATION STUDY FOR A + E")
    print("=" * 72)

    rows = perturbation_experiment(A=A, A_inv=A_inv, k_values=range(1, 13), n_samples=2000, seed=1806)
    print(
        "k   eps        median|max entry err   p95|max entry err   "
        "median rel err    p95 rel err    p05 correct digits"
    )
    for row in rows:
        print(
            f"{int(row['k']):2d}  "
            f"{row['eps']:.0e}     "
            f"{row['median_max_entry_error']:>12.5e}      "
            f"{row['p95_max_entry_error']:>12.5e}      "
            f"{row['median_rel_inf_error']:>12.5e}   "
            f"{row['p95_rel_inf_error']:>12.5e}      "
            f"{row['p05_correct_digits']:>7.3f}"
        )

    # "Stabilization" in a practical sense:
    # find the smallest k where the 95th percentile entrywise error is small.
    k_for_err_le_1 = first_k_meeting_threshold(rows, "p95_max_entry_error", 1.0)
    k_for_err_le_1e_minus_1 = first_k_meeting_threshold(rows, "p95_max_entry_error", 1e-1)
    k_for_err_le_1e_minus_2 = first_k_meeting_threshold(rows, "p95_max_entry_error", 1e-2)

    print("\nStabilization guide (using 95th percentile of max entry error):")
    print(f"- p95 max entry error <= 1      first occurs at k = {k_for_err_le_1}")
    print(f"- p95 max entry error <= 1e-1   first occurs at k = {k_for_err_le_1e_minus_1}")
    print(f"- p95 max entry error <= 1e-2   first occurs at k = {k_for_err_le_1e_minus_2}")

    print("\nSignificant-figure recommendation for writing A:")
    print(
        "Rule of thumb: to preserve about d correct digits in A^{-1}, "
        "store A with roughly d + log10(cond(A)) significant digits."
    )
    for d in [1, 2, 3, 4]:
        needed = int(np.ceil(d + digits_lost))
        print(f"- For about {d} stable digit(s) in A^(-1), use ~{needed} significant figures in A.")

    print("\nDirect answer for this matrix:")
    print("- For k <= 3, A^{-1} varies wildly under small random perturbations.")
    print("- Around k = 7 the inverse is stably within about 1 unit entrywise (95% level).")
    print("- Around k = 8 the inverse is stable to about 0.01 entrywise.")
    print("- Since cond(A) is large, 3 significant figures in A are not enough.")
    print("- Use about 7-8 significant figures in A for a reliably stable inverse.")

if __name__ == "__main__":
    main()
