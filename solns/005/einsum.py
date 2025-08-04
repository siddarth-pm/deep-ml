import numpy as np


def scalar_multiply(matrix: list[list[int | float]], scalar: int | float) -> list[list[int | float]]:
    result = np.einsum('ij,->ij', matrix, scalar)
    return result
