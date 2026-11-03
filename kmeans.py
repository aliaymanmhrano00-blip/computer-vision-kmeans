"""
K-Means Clustering on Synthetic Points
Computer Vision Task

Run:
    python kmeans.py

Optional arguments:
    python kmeans.py --k 3 --points 150 --seed 42
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def create_points(seed: int = 42, points_per_cluster: int = 50) -> np.ndarray:
    """Create synthetic 2D points around three known centers."""
    rng = np.random.default_rng(seed)

    centers = np.array([
        [2.0, 2.0],
        [7.0, 3.0],
        [4.5, 8.0],
    ])

    clusters = [
        rng.normal(loc=center, scale=0.9, size=(points_per_cluster, 2))
        for center in centers
    ]

    return np.vstack(clusters)


def initialize_centroids(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Choose k different points as the initial centroids."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if k > len(points):
        raise ValueError("k cannot be greater than the number of points")

    indices = rng.choice(len(points), size=k, replace=False)
    return points[indices].copy()


def assign_clusters(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign every point to its nearest centroid."""
    distances = np.linalg.norm(points[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(
    points: np.ndarray,
    labels: np.ndarray,
    old_centroids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Recalculate centroids; reinitialize an empty cluster if needed."""
    new_centroids = []

    for cluster_id in range(len(old_centroids)):
        cluster_points = points[labels == cluster_id]

        if len(cluster_points) == 0:
            # Prevent an empty cluster from breaking the algorithm.
            new_centroids.append(points[rng.integers(0, len(points))])
        else:
            new_centroids.append(cluster_points.mean(axis=0))

    return np.array(new_centroids)


def kmeans(
    points: np.ndarray,
    k: int,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run K-Means and return labels, centroids, and iteration count."""
    rng = np.random.default_rng(seed)
    centroids = initialize_centroids(points, k, rng)

    for iteration in range(1, max_iterations + 1):
        labels = assign_clusters(points, centroids)
        new_centroids = update_centroids(points, labels, centroids, rng)

        centroid_shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        if centroid_shift < tolerance:
            # Reassign once more using the final centroid positions.
            labels = assign_clusters(points, centroids)
            return labels, centroids, iteration

    labels = assign_clusters(points, centroids)
    return labels, centroids, max_iterations


def plot_results(
    points: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    output_path: Path,
) -> None:
    """Save a visualization of clustered points and centroids."""
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        cmap="viridis",
        s=45,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.4,
    )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="red",
        marker="X",
        s=240,
        edgecolors="black",
        linewidths=1.2,
        label="Centroids",
    )

    plt.title("K-Means Clustering on Synthetic Points")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="K-Means clustering from scratch")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters")
    parser.add_argument("--points", type=int, default=50, help="Points generated per original cluster")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum K-Means iterations")
    args = parser.parse_args()

    if args.points <= 0:
        raise ValueError("--points must be greater than zero")

    points = create_points(seed=args.seed, points_per_cluster=args.points)
    labels, centroids, iterations = kmeans(
        points,
        k=args.k,
        max_iterations=args.max_iterations,
        seed=args.seed,
    )

    output_path = Path(__file__).with_name("kmeans_result.png")
    plot_results(points, labels, centroids, output_path)

    print(f"K-Means finished in {iterations} iteration(s).")
    print("Final centroids:")
    print(np.round(centroids, 3))
    print(f"Result image saved to: {output_path}")


if __name__ == "__main__":
    main()
