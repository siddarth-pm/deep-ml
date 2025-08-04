import numpy as np


def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == 'row':
        means = np.einsum('ij->i', matrix)/len(matrix[0])
    else:
        means = np.einsum('ij->j', matrix)/len(matrix)
    return means
