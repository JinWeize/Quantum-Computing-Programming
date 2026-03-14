import numpy as np
import math

A = np.array([[1, -1j, 0, 1],
                    [1j, 1, 0, 1j],
                    [0, 0, 0, 0],
                    [1, -1j, 0, 1]
                    ])

A = (1 / 3) * A
print(A)
eigen_values, eigen_vectors = np.linalg.eig(A)
print(eigen_values)
print("\n",eigen_vectors)

psi = np.array([1/math.sqrt(2), 1/2, 0, -1/2])

print(psi.conj().T @ A @ psi)