import numpy as np


def transform_matrix(A: list[list[int | float]], T: list[list[int | float]], S: list[list[int | float]]) -> list[list[int | float]]:
    # verify if T is invertible
    if np.linalg.det(T) == 0:
        return -1
    # verify if S is invertible
    if np.linalg.det(S) == 0:
        return -1
    # invert T
    inv_T = np.linalg.inv(T)
    # multiply matrices
    transformed_matrix = inv_T @ A @ S
    # einsum version
    transformed_matrix = np.einsum('ij,jk,kl->il', inv_T, A, S)
    return transformed_matrix
