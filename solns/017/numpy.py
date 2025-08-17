import numpy as np


def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
    X = np.asarray(points, dtype=np.float32)  # n, d
    C = np.asarray(initial_centroids, dtype=np.float32)  # k, d
    for _ in range(max_iterations):
        sub = X[:, None, :] - C[None, :, :]  # (n, k, d)
        square = sub**2  # (n, k, d)
        dists_sq = np.sum(square, axis=2)  # (n, k)

        labels = np.argmin(dists_sq, axis=1)  # (n,)
        new_C = np.zeros_like(C)
        for i in range(k):  # iterate over centroids
            cur_points = X[labels == i]  # matching points for this centroid
            new_C[i] = cur_points.mean(axis=0)  # assign new centroid
        C = new_C  # update
    return [tuple(np.round(c, 4)) for c in C]
