import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n = len(vectors)
    cov = [[0] * n for _ in range(n)]
    means = [None] * n
    for i in range(n):
        means[i] = np.mean(vectors[i])
        for j in range(min(i+1, n)):
            cov[i][j] = covariance(vectors[i], vectors[j], means[i], means[j], n)
            cov[j][i] = cov[i][j] 
	return cov

def covariance(vec1: list[float], vec2: list[float], mean1: float, mean2: float, n: int):
    ans = 0
    for i in range(n):
        ans += (vec1[i] - mean1) * (vec2[i] - mean2)
    return ans/(n-1)
