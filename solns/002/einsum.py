import numpy as np
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    return np.einsum("ij->ji", a)