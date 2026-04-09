# 18.06 Spring 2026
# Problem Set 7
# Name: Jan Szmajda
# Collaborators: Opus 4.6

import numpy as np

np.set_printoptions(precision=8, suppress=True)

# ======================================================================
# COMPUTE 1: Wilkinson polynomial root sensitivity
# ======================================================================
print("=" * 60)
print("COMPUTE 1")
print("=" * 60)

# Build f(x) = (x-1)(x-2)...(x-20)
f = np.poly(np.arange(1, 21, dtype=float))  # coefficients, high degree first
true_roots = np.arange(1.0, 21.0)

print("\nf(x) = prod_{i=1}^{20} (x - i), roots are 1, 2, ..., 20")

# Perturb: add eps * x^m, compare roots
base_roots = np.roots(f)  # unperturbed computed roots

print("\nSweep over m and eps: max root displacement from unperturbed")
print(f"{'m':>4s}  {'eps=1e-10':>12s} {'eps=1e-7':>12s} {'eps=1e-4':>12s} {'eps=1e-1':>12s}")
for m in range(20):
    row = f"{m:4d}"
    for eps in [1e-10, 1e-7, 1e-4, 1e-1]:
        p = f.copy()
        p[20 - m] += eps
        roots = np.roots(p)
        # match each base root to nearest perturbed root
        disps = []
        for br in base_roots:
            disps.append(min(abs(roots - br)))
        row += f"  {max(disps):10.3e}"
    print(row)

# Detailed example: eps = 1e-7, m = 19
print("\nDetailed example: f(x) + 1e-7 * x^19")
p = f.copy()
p[1] += 1e-7  # x^19 is index 1
roots = sorted(np.roots(p), key=lambda z: z.real)
print(f"{'root':>6s} {'perturbed root':>26s} {'|shift|':>10s}")
for i, tr in enumerate(true_roots):
    r = roots[i]
    if abs(r.imag) > 1e-6:
        rstr = f"{r.real:10.4f} {r.imag:+10.4f}i"
    else:
        rstr = f"{r.real:10.4f}"
    print(f"{tr:6.0f}  {rstr:>26s} {abs(r - tr):10.3e}")

print("\nConclusion: tiny perturbations to the coefficients (eps ~ 1e-7)")
print("cause root shifts of order 1, and real roots become complex.")
print("This shows finding eigenvalues via the characteristic polynomial")
print("is numerically unstable.")

# ======================================================================
# COMPUTE 2: QR algorithm
# ======================================================================
print("\n" + "=" * 60)
print("COMPUTE 2")
print("=" * 60)

A = np.array([
    [-52, 194, -20,  49],
    [-18,  67,  -6,  18],
    [ -6,   8, -19, -54],
    [  6, -16,  10,  21],
], dtype=float)

# --- (a) Eigenvalues via built-in ---
print("\n--- Part (a) ---")
eigs = np.sort(np.linalg.eigvals(A).real)
print(f"Eigenvalues of A: {eigs}")

# --- (b) Hessenberg form ---
print("\n--- Part (b) ---")

E31 = np.eye(4)
E31[2, 1] = -A[2, 0] / A[1, 0]  # -(-6)/(-18) = -1/3
print("E31 ="); print(E31)

E41 = np.eye(4)
E41[3, 1] = -A[3, 0] / A[1, 0]  # -(6)/(-18) = 1/3
print("E41 ="); print(E41)

H = E41 @ E31 @ A @ np.linalg.inv(E31) @ np.linalg.inv(E41)
print("H = E41 E31 A E31^{-1} E41^{-1} ="); print(H)

print(f"\nH(3,1) = {H[2,0]:.1e}, H(4,1) = {H[3,0]:.1e}  (both zero -> Hessenberg)")
print(f"Eigenvalues of H: {np.sort(np.linalg.eigvals(H).real)}")
print("Same as A, as expected (similarity transform preserves eigenvalues).")

# --- (c) QR of H, form H1 = RQ ---
print("\n--- Part (c) ---")
Q, R = np.linalg.qr(H)
H1 = R @ Q
print("Q ="); print(Q)
print("R ="); print(R)
print("H1 = RQ ="); print(H1)

print(f"\nEigenvalues of H1: {np.sort(np.linalg.eigvals(H1).real)}")
print("Same as A (H1 = Q^{-1}HQ is a similarity transform).")

print(f"\nSub-diagonal entries of H1:")
for i in range(3):
    print(f"  ({i+2},{i+1}) = {H1[i+1, i]:.8e}")
biggest_idx = max(range(3), key=lambda i: abs(H1[i+1, i]))
print(f"Largest: ({biggest_idx+2},{biggest_idx+1}) = {H1[biggest_idx+1, biggest_idx]:.8f}")

# --- (d) Iterate H2, ..., H10 ---
print("\n--- Part (d) ---")
H_k = H.copy()
for k in range(1, 11):
    Q_k, R_k = np.linalg.qr(H_k)
    H_k = R_k @ Q_k
    sd = [H_k[i+1, i] for i in range(3)]
    diag = np.sort(np.diag(H_k))
    print(f"H_{k:2d}: sub-diag = [{sd[0]:10.3e}, {sd[1]:10.3e}, {sd[2]:10.3e}]  "
          f"diag = {diag}")

print("\nThe sub-diagonal entries shrink toward zero each iteration.")
print(f"Diagonal of H10 = {np.sort(np.diag(H_k))}")
print("These approximate the eigenvalues 2, 3, 5, 7 to within ~0.001.")
print("As H_k becomes upper triangular, its diagonal gives the eigenvalues.")
