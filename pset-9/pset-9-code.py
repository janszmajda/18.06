# 18.06 Spring 2026
# Problem Set 9
# Name: Jan Szmajda
# Collaborators: Codex 5.3

from __future__ import annotations

import gzip
import os
import struct
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-18.06-pset9")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n_images, n_rows, n_cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file.")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n_images, n_rows, n_cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file.")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n_labels)


def load_mnist_test() -> tuple[np.ndarray, np.ndarray]:
    image_path = DATA_DIR / "t10k-images-idx3-ubyte.gz"
    label_path = DATA_DIR / "t10k-labels-idx1-ubyte.gz"

    missing = [path for path in [image_path, label_path] if not path.exists()]
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            "Missing MNIST files: "
            f"{missing_names}. Put the Kaggle/Yann LeCun IDX gzip files in {DATA_DIR}."
        )

    return read_idx_images(image_path), read_idx_labels(label_path)


def flatten_images(images: np.ndarray) -> np.ndarray:
    return images.reshape(images.shape[0], -1).astype(float) / 255.0


def one_example_per_digit(labels: np.ndarray) -> list[int]:
    indices: list[int] = []
    for digit in range(10):
        matches = np.flatnonzero(labels == digit)
        if matches.size == 0:
            raise ValueError(f"No image found for digit {digit}.")
        indices.append(int(matches[0]))
    return indices


def save_reconstruction_grid(
    X: np.ndarray,
    labels: np.ndarray,
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    k_values: list[int],
    output_path: Path,
) -> None:
    sample_indices = one_example_per_digit(labels)
    n_rows = len(k_values) + 1
    n_cols = len(sample_indices)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.15 * n_cols, 1.25 * n_rows))

    for j, idx in enumerate(sample_indices):
        ax = axes[0, j]
        ax.imshow(X[idx].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(str(labels[idx]), fontsize=9)
        ax.axis("off")
    axes[0, 0].set_ylabel("orig", fontsize=9)

    for i, k in enumerate(k_values, start=1):
        reconstruction = (U[sample_indices, :k] * s[:k]) @ Vt[:k, :]
        reconstruction = np.clip(reconstruction, 0.0, 1.0)
        for j in range(n_cols):
            ax = axes[i, j]
            ax.imshow(reconstruction[j].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        axes[i, 0].set_ylabel(f"k={k}", fontsize=9)

    fig.suptitle("Rank-k SVD reconstructions of one MNIST test image per digit", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def retained_energy(s: np.ndarray, k: int) -> float:
    return float(np.sum(s[:k] ** 2) / np.sum(s**2))


def centroid_predictions(
    X: np.ndarray,
    labels: np.ndarray,
    Vt: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project every image into the common rank-k right-singular-vector subspace,
    average by digit to make ten cluster centers, then classify by nearest center.
    """
    Vk = Vt[:k, :]
    projected = X @ Vk.T

    centers = np.zeros((10, k), dtype=float)
    for digit in range(10):
        centers[digit] = projected[labels == digit].mean(axis=0)

    diff = projected[:, None, :] - centers[None, :, :]
    distances = np.sum(diff * diff, axis=2)
    predictions = np.argmin(distances, axis=1)
    return predictions, centers


def accuracy_by_k(
    X: np.ndarray,
    labels: np.ndarray,
    Vt: np.ndarray,
    k_values: list[int],
) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for k in k_values:
        predictions, _ = centroid_predictions(X, labels, Vt, k)
        rows.append((k, float(np.mean(predictions == labels))))
    return rows


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    C = np.zeros((10, 10), dtype=int)
    for actual, predicted in zip(labels, predictions):
        C[int(actual), int(predicted)] += 1
    return C


def print_compute_1() -> None:
    images, labels = load_mnist_test()
    X = flatten_images(images)

    print("=" * 72)
    print("COMPUTE 1: MNIST AND SVD")
    print("=" * 72)
    print(f"Loaded test images: {images.shape}")
    print(f"Flattened matrix X shape: {X.shape}")

    print("\nComputing full SVD of the 10000 x 784 image matrix...")
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    print("\n(a) Rank-k approximation")
    print(" k    retained Frobenius energy    relative Frobenius error")
    for k in [1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]:
        energy = retained_energy(s, k)
        rel_error = np.sqrt(max(0.0, 1.0 - energy))
        print(f"{k:3d}          {energy:10.4f}                  {rel_error:10.4f}")

    reconstruction_path = OUT_DIR / "mnist_rank_k_reconstructions.png"
    save_reconstruction_grid(
        X=X,
        labels=labels,
        U=U,
        s=s,
        Vt=Vt,
        k_values=[5, 10, 20, 40, 80],
        output_path=reconstruction_path,
    )

    print(f"\nSaved reconstruction grid to: {reconstruction_path}")
    print("Visual conclusion from the grid:")
    print("- k around 20 is usually enough to recognize the digit reliably.")
    print("- k around 40 makes the reconstructions clearly match the originals.")
    print("- k around 80 is close to the original image for this purpose.")

    print("\n(b) Nearest projected cluster center")
    k_values = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 400, 784]
    acc_rows = accuracy_by_k(X=X, labels=labels, Vt=Vt, k_values=k_values)
    print(" k    accuracy")
    for k, acc in acc_rows:
        print(f"{k:3d}    {100 * acc:8.3f}%")

    chosen_k = 40
    predictions, _ = centroid_predictions(X=X, labels=labels, Vt=Vt, k=chosen_k)
    C = confusion_matrix(labels, predictions)
    print(f"\nUsing k = {chosen_k}, accuracy = {100 * np.mean(predictions == labels):.3f}%")
    print("Confusion matrix rows are true digits and columns are predicted digits:")
    print(C)

    best_k, best_acc = max(acc_rows, key=lambda row: row[1])
    print("\nAccuracy summary:")
    print(f"- Best value in this sweep: k = {best_k}, accuracy = {100 * best_acc:.3f}%.")
    print("- Accuracy rises quickly through k = 10-40, then mostly levels off.")
    print("- Keeping many more singular vectors adds little because nearest-centroid")
    print("  classification is limited by overlap between the digit classes.")


def singular_value_trials(
    m: int,
    n: int,
    trials: int,
    distribution: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    singular_values = np.empty((trials, min(m, n)), dtype=float)

    for t in range(trials):
        if distribution == "uniform":
            A = rng.random((m, n))
        elif distribution == "gaussian":
            A = rng.standard_normal((m, n))
        else:
            raise ValueError("distribution must be 'uniform' or 'gaussian'")

        singular_values[t] = np.linalg.svd(A, compute_uv=False)

    return singular_values


def save_singular_value_plot(
    uniform_s: np.ndarray,
    gaussian_s: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    idx = np.arange(1, uniform_s.shape[1] + 1)
    axes[0].plot(idx, uniform_s.mean(axis=0), "o-", label="Uniform[0,1]")
    axes[0].plot(idx, gaussian_s.mean(axis=0), "o-", label="N(0,1)")
    axes[0].set_xlabel("singular value index")
    axes[0].set_ylabel("mean singular value")
    axes[0].set_title("Mean ordered singular values")
    axes[0].legend()

    axes[1].hist(
        uniform_s[:, 1:].ravel(),
        bins=35,
        alpha=0.65,
        density=True,
        label="Uniform[0,1], except largest",
    )
    axes[1].hist(
        gaussian_s.ravel(),
        bins=35,
        alpha=0.65,
        density=True,
        label="N(0,1), all",
    )
    axes[1].axvline(uniform_s[:, 0].mean(), color="tab:blue", linestyle="--", label="uniform largest mean")
    axes[1].set_xlabel("singular value")
    axes[1].set_ylabel("density")
    axes[1].set_title("Bulk singular values")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_compute_2() -> None:
    print("\n" + "=" * 72)
    print("COMPUTE 2: SINGULAR VALUES OF RANDOM 20 x 40 MATRICES")
    print("=" * 72)

    m, n = 20, 40
    trials = 2000
    uniform_s = singular_value_trials(m=m, n=n, trials=trials, distribution="uniform", seed=2026)
    gaussian_s = singular_value_trials(m=m, n=n, trials=trials, distribution="gaussian", seed=2027)

    print(f"matrix_shape = {m} x {n}")
    print(f"trials = {trials}")

    print("\nSummary of ordered singular values:")
    print("distribution       mean s1    mean s2    mean s20   median s1  median s20")
    print(
        f"Uniform[0,1]      "
        f"{uniform_s[:, 0].mean():8.4f}   "
        f"{uniform_s[:, 1].mean():8.4f}   "
        f"{uniform_s[:, -1].mean():8.4f}   "
        f"{np.median(uniform_s[:, 0]):8.4f}   "
        f"{np.median(uniform_s[:, -1]):8.4f}"
    )
    print(
        f"N(0,1)            "
        f"{gaussian_s[:, 0].mean():8.4f}   "
        f"{gaussian_s[:, 1].mean():8.4f}   "
        f"{gaussian_s[:, -1].mean():8.4f}   "
        f"{np.median(gaussian_s[:, 0]):8.4f}   "
        f"{np.median(gaussian_s[:, -1]):8.4f}"
    )

    print("\nMean ordered singular values:")
    print(" i    Uniform[0,1]      N(0,1)")
    for i in range(20):
        print(f"{i + 1:2d}      {uniform_s[:, i].mean():10.4f}   {gaussian_s[:, i].mean():10.4f}")

    plot_path = OUT_DIR / "random_matrix_singular_values.png"
    save_singular_value_plot(uniform_s=uniform_s, gaussian_s=gaussian_s, output_path=plot_path)
    print(f"\nSaved singular-value comparison plot to: {plot_path}")

    print("\nConclusion:")
    print("- Uniform[0,1] matrices have a large positive mean, so s1 is a rank-one")
    print("  spike near 0.5*sqrt(20*40) = 14.1421.")
    print("- After that spike, the remaining uniform singular values are small because")
    print("  the entry variance is only 1/12.")
    print("- Gaussian N(0,1) matrices are centered and have variance 1, so there is no")
    print("  mean spike; instead the whole singular-value bulk is much larger.")


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    OUT_DIR.mkdir(exist_ok=True)
    print_compute_1()
    print_compute_2()


if __name__ == "__main__":
    main()
