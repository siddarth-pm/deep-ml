import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	m = b.shape[0]
	x = np.zeros(m)
	for _ in range(n):
		cur_x = np.empty_like(x)
		for i in range(m):
			res = np.einsum('j,j->', A[i], x) - A[i][i] * x[i]
			cur_x[i] = (1/A[i][i]) * (b[i] - res)
		x = np.round(cur_x, 4)
	return x
