import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
    if len(a[0]) * len(a) != new_shape[0] * new_shape[1]:
        return []
    reshaped_matrix = np.reshape(a, new_shape)
    return reshaped_matrix