# 18.06 Spring 2026
# Problem Set 1
# Name: Jan Szmajda
# Collaborators: SWE-1.5 Coding Model

import time
import numpy as np
from scipy.linalg import lu_factor
import matplotlib.pyplot as plt

## Question 1 (Compute 1 – 20 points)

# This problem was part of a 2026 Mystery Hunt puzzle.
# Suppose that each letter has a value, and the value of a word
# is the sum of the constituent letters.

# We are given the values of the following words:

# COFFEE   59
# QUICHE   55
# APPLE    44
# PIE      33
# PEA      28
# PEPPER   25
# PEAR     22
# CHEESE   19
# KIWI     17
# LEMON    16
# MELON    16
# ONION    12
# PAPAYA    9

# PIZZA     8
# BREAD     0
# GRAPE    -3
# BACON    -6
# COOKIE   -9
# OREO    -10
# TOMATO  -19
# CORN    -27
# CARROT  -28
# DONUT   -29
# EGG     -36
# CHERRY  -38
# MUSHROOM -50

# Find the values of:

# FONDUE
# WATER
# CROQUE MONSIEUR
# POPSICLE
# BOBA
# ONIGIRI
# PLUM
# MAYONNAISE

# CHEESE
# SELTZER
# CROQUE MADAME
# ICE CREAM BAR
# TAPIOCA
# UMEBOSHI
# PRUNE
# REMOULADE

# Converting numbers into letters using A=1, ..., Z=26,
# what phrase does this spell?

# Set up the system: Ax = b
# where A is the coefficient matrix (words × letters)
# x is the vector of letter values (26 × 1)
# b is the vector of word values

# Map letters to indices (A=0, B=1, ..., Z=25)
letter_to_idx = {chr(ord('A') + i): i for i in range(26)}

# Given words and their values
given_words = [
    ("COFFEE", 59),
    ("QUICHE", 55),
    ("APPLE", 44),
    ("PIE", 33),
    ("PEA", 28),
    ("PEPPER", 25),
    ("PEAR", 22),
    ("CHEESE", 19),
    ("KIWI", 17),
    ("LEMON", 16),
    ("MELON", 16),
    ("ONION", 12),
    ("PAPAYA", 9),
    ("PIZZA", 8),
    ("BREAD", 0),
    ("GRAPE", -3),
    ("BACON", -6),
    ("COOKIE", -9),
    ("OREO", -10),
    ("TOMATO", -19),
    ("CORN", -27),
    ("CARROT", -28),
    ("DONUT", -29),
    ("EGG", -36),
    ("CHERRY", -38),
    ("MUSHROOM", -50)
]

# Create coefficient matrix A and vector b
A = []
b = []

for word, value in given_words:
    row = [0] * 26
    for letter in word:
        row[letter_to_idx[letter]] += 1
    A.append(row)
    b.append(value)

A = np.array(A)
b = np.array(b)

print(f"Matrix A shape: {A.shape}")
print(f"Vector b shape: {b.shape}")

# Solve the system using least squares (since we have 26 unknowns and 25 equations)
x, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)

print(f"Rank of A: {rank}")
print(f"Residuals: {residuals}")

# Create dictionary of letter values
letter_values = {}
for i, letter in enumerate(chr(ord('A') + j) for j in range(26)):
    letter_values[letter] = x[i]

print("\nLetter values:")
for letter, value in sorted(letter_values.items()):
    print(f"{letter}: {value:.2f}")

# Words to find values for
target_words = [
    "FONDUE", "WATER", "CROQUEMONSIEUR", "POPSICLE", "BOBA", "ONIGIRI",
    "PLUM", "MAYONNAISE", "CHEESE", "SELTZER", "CROQUEMADAME",
    "ICECREAMBAR", "TAPIOCA", "UMEBOSHI", "PRUNE", "REMOULADE"
]

print("\nTarget word values:")
word_values = []
for word in target_words:
    value = sum(letter_values[letter] for letter in word)
    word_values.append(value)
    print(f"{word}: {value:.2f}")

# Convert to integers (rounding to nearest integer)
word_values_int = [round(val) for val in word_values]
print("\nRounded values:")
for word, val in zip(target_words, word_values_int):
    print(f"{word}: {val}")

# Convert numbers to letters (A=1, B=2, ..., Z=26)
print("\nConverting to letters:")
phrase = ""
for val in word_values_int:
    if 1 <= val <= 26:
        phrase += chr(ord('A') + val - 1)
    else:
        phrase += f"[{val}]"

print(f"Phrase: {phrase}")


## Question 2 (Compute 1 – 10 points)

# Determine how long LU-decomposition for a random square n × n matrix
# takes as a function of n in the system you are using.
# Choose n large enough so that the dependence on n is clear
# (very small n will all take about the same time).

# Your answer should take the form:
#     T(n) = C * n^alpha
# Try to find the best values of C and alpha that fit your measured data.

# Test different matrix sizes (avoid tiny n where overhead dominates)
sizes = [200, 300, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
times = []
num_trials = 3  # Average over multiple trials for more reliable timing

rng = np.random.default_rng(0)

# Warm-up to reduce one-time overhead effects
lu_factor(rng.standard_normal((200, 200)))

print("Measuring LU-decomposition times (averaging over {} trials):".format(num_trials))
for n in sizes:
    trial_times = []

    # Pre-generate matrices so timing excludes random number generation
    matrices = [rng.standard_normal((n, n)) for _ in range(num_trials)]

    for A in matrices:
        start_time = time.perf_counter()
        lu_factor(A)
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        trial_times.append(elapsed_time)

    # Average over trials
    avg_time = np.mean(trial_times)
    times.append(avg_time)
    print(f"n={n}: {avg_time:.6f} seconds (std: {np.std(trial_times):.6f})")

print("\nSize vs Time data:")
for n, t in zip(sizes, times):
    print(f"({n}, {t:.6f})")

# Fit the data to T(n) = C * n^alpha using log-log regression
log_sizes = np.log(sizes)
log_times = np.log(times)

# Linear regression in log-log space
coeffs = np.polyfit(log_sizes, log_times, 1)
alpha = coeffs[0]
log_C = coeffs[1]
C = np.exp(log_C)

print(f"\nFitted parameters:")
print(f"alpha (exponent): {alpha:.4f}")
print(f"C (constant): {C:.8f}")
print(f"T(n) = {C:.8f} * n^{alpha:.4f}")

# Calculate R-squared to assess fit quality
predicted_log_times = alpha * log_sizes + log_C
ss_res = np.sum((log_times - predicted_log_times) ** 2)
ss_tot = np.sum((log_times - np.mean(log_times)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"R-squared: {r_squared:.6f}")

# Generate fitted curve for comparison
fitted_times = [C * n**alpha for n in sizes]

# Save data for MATLAB/Octave plotting
np.savetxt(
    "q2_lu_timing.csv",
    np.column_stack([sizes, times]),
    delimiter=",",
    header="n,time_sec",
    comments=""
)

print("\n" + "="*60)
print("QUESTION 2 - LU DECOMPOSITION TIMING RESULTS")
print("="*60)

print(f"\nFitted model: T(n) = {C:.2e} * n^{alpha:.4f}")
print(f"Theoretical complexity: O(n^3)")
print(f"Measured exponent: {alpha:.4f}")
print(f"R-squared: {r_squared:.6f} (higher is better, max is 1.0)")

print(f"\nInterpretation:")
print(f"- The exponent α = {alpha:.4f} is close to the theoretical value of 3")
print(f"- The constant C = {C:.2e} represents the base computational overhead")
print(f"- High R-squared ({r_squared:.6f}) indicates good fit to the power law")

print(f"\nFor practical use:")
print(f"- A 1000×1000 matrix takes ~{times[sizes.index(1000)]:.4f} seconds")
print(f"- A 2000×2000 matrix takes ~{times[sizes.index(2000)]:.4f} seconds")
print(f"- Doubling matrix size increases time by ~{2**alpha:.1f}x")

# Python plot (log-log) so the figure appears directly when running this script
plt.figure(figsize=(8, 5))
plt.loglog(sizes, times, "o", label="Measured")
plt.loglog(sizes, fitted_times, "-", label=rf"Fit: $T(n)={C:.2e}n^{{{alpha:.3f}}}$")
plt.xlabel("n")
plt.ylabel("Time (seconds)")
plt.title("LU Decomposition Timing")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("q2_lu_plot.png", dpi=200)
plt.show()
