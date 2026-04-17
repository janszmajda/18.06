# 18.06 Spring 2026
# Problem Set 8 (bonus)
# Name: Jan Szmajda
# Collaborators: Opus 4.7
# System used: Python/NumPy

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(18_06)

# Symmetric matrix: upper-triangular entries (incl. diagonal) iid N(mu, 1),
# then mirror.  Each entry has variance 1 as the problem asks.
def random_sym(n, mu_diag=0.0, mu_off=0.0):
    U = rng.standard_normal((n, n))
    S = np.triu(U, 1) + np.triu(U, 1).T + np.diag(np.diag(U))
    S += mu_off * (np.ones((n, n)) - np.eye(n)) + mu_diag * np.eye(n)
    return S

def lam_min(S):
    return np.linalg.eigvalsh(S)[0]

# ======================================================================
# Part (a): mean 0, variance 1
# ======================================================================
print("=" * 60)
print("Part (a): mean 0, variance 1")
print("=" * 60)

ns = [2, 3, 5, 10, 20, 50, 100, 200]
N = 3000

print(f"\n{'n':>4s} {'E[lam_min]':>12s} {'-2*sqrt(n)':>12s} {'P(pos def)':>12s}")
stats = {}
for n in ns:
    lm = np.array([lam_min(random_sym(n)) for _ in range(N)])
    stats[n] = lm
    print(f"{n:4d} {lm.mean():12.3f} {-2*np.sqrt(n):12.3f} {(lm > 0).mean():12.4f}")

# Histograms
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, n in zip(axes.ravel(), ns):
    ax.hist(stats[n], bins=30, color="steelblue", edgecolor="white")
    ax.axvline(-2 * np.sqrt(n), color="red", ls="--")
    ax.set_title(f"n = {n}")
fig.suptitle("Smallest eigenvalue (red dashed = -2*sqrt(n))")
fig.tight_layout()
fig.savefig("/Users/janszmajda/Documents/18.06/pset-8/lambda_min_hist.png", dpi=110)

# P(PD) for small n with more trials
print("\nP(positive definite), 100,000 trials:")
for n in [1, 2, 3, 4, 5, 6]:
    pd = sum(lam_min(random_sym(n)) > 0 for _ in range(100_000))
    print(f"  n={n}: {pd/100_000:.5f}")

print("\n(a) lam_min ~ -2*sqrt(n) (Wigner semicircle edge).  P(PD) halves at")
print("each n: 0.50, 0.12, 0.012, 0.0006, ~0, ~0 for n = 1..6.  Random")
print("symmetric Gaussians are essentially never PD once n > ~5.")

# ======================================================================
# Part (b): all entries mean mu, variance 1
# ======================================================================
print("\n" + "=" * 60)
print("Part (b): all entries mean mu, variance 1")
print("=" * 60)

# S = S0 + mu*J where J = ones(n,n) is rank 1 with eigenvalues n, 0,...,0.
# So mu adds one spike eigenvalue ~ mu*n on top; bulk (incl. lam_min) unchanged.

print(f"\n{'n':>4s} {'mu':>5s} {'E[lam_min]':>12s} {'E[lam_max]':>12s} {'mu*n':>8s}")
for n in [10, 50]:
    for mu in [0.0, 0.5, 2.0]:
        lmins = np.empty(1000)
        lmaxs = np.empty(1000)
        for t in range(1000):
            w = np.linalg.eigvalsh(random_sym(n, mu, mu))
            lmins[t], lmaxs[t] = w[0], w[-1]
        print(f"{n:4d} {mu:5.1f} {lmins.mean():12.3f} {lmaxs.mean():12.3f} {mu*n:8.1f}")

print("\n(b) The mean matrix mu*J has rank 1 (eigenvalues mu*n, 0,...,0), so it")
print("only creates a single 'spike' near mu*n at the top of the spectrum.")
print("lam_min is essentially independent of mu -- still ~ -2*sqrt(n).")
print("Making mu large does NOT help PD; it only inflates lam_max.")

# ======================================================================
# Part (c): diagonal mean mu1, off-diagonal mean mu2
# ======================================================================
print("\n" + "=" * 60)
print("Part (c): diagonal mean mu1, off-diagonal mean mu2")
print("=" * 60)

# S = S0 + (mu1 - mu2)*I + mu2*J.  Mean part has one eigenvalue (n-1)*mu2 + mu1
# and (n-1) copies of mu1 - mu2.  For PD we need BOTH
#   (n-1)*mu2 + mu1 > 0      (spike)
#   mu1 - mu2 > 2*sqrt(n)    (bulk shifted past semicircle edge)

n = 5
mu1_vals = np.linspace(-2, 8, 11)
mu2_vals = np.linspace(-2, 4, 13)
P = np.zeros((len(mu1_vals), len(mu2_vals)))
for i, mu1 in enumerate(mu1_vals):
    for j, mu2 in enumerate(mu2_vals):
        P[i, j] = np.mean([lam_min(random_sym(n, mu1, mu2)) > 0 for _ in range(1500)])

print(f"\nP(PD) grid for n = {n}:")
print("mu1\\mu2 " + "".join(f"{m:6.1f}" for m in mu2_vals))
for i, mu1 in enumerate(mu1_vals):
    print(f"{mu1:7.1f} " + "".join(f"{P[i,j]:6.2f}" for j in range(len(mu2_vals))))

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(P, origin="lower", aspect="auto",
               extent=[mu2_vals[0], mu2_vals[-1], mu1_vals[0], mu1_vals[-1]],
               cmap="viridis", vmin=0, vmax=1)
m2 = np.linspace(mu2_vals[0], mu2_vals[-1], 100)
ax.plot(m2, -(n - 1) * m2, "r--", label="spike = 0")
ax.plot(m2, m2 + 2 * np.sqrt(n), "w--", label=f"bulk edge: mu1-mu2 = 2*sqrt(n)")
ax.set_xlabel("mu2 (off-diagonal)"); ax.set_ylabel("mu1 (diagonal)")
ax.set_title(f"P(positive definite), n = {n}")
ax.set_xlim(mu2_vals[0], mu2_vals[-1]); ax.set_ylim(mu1_vals[0], mu1_vals[-1])
ax.legend(loc="lower right"); fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig("/Users/janszmajda/Documents/18.06/pset-8/p_pd_heatmap.png", dpi=110)

print("\n(c) The mean matrix mu1*I + mu2*(J-I) has eigenvalues (n-1)*mu2 + mu1")
print("(once) and mu1 - mu2 (n-1 times).  PD needs BOTH (n-1)*mu2 + mu1 > 0")
print("AND mu1 - mu2 > 2*sqrt(n).  The second condition dominates: only")
print("diagonal dominance (mu1 much larger than mu2) makes PD likely.")
