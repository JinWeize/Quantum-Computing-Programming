from pyqpanda3.transpilation import decompose
import numpy as np

mat = 1/2 * np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])
print(mat)
f = decompose(mat)
print(f)