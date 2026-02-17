import numpy as np


ket0 = np.array([1, 0, 0], dtype=complex)
ket1 = np.array([0, 1, 0], dtype=complex)
ket2 = np.array([0, 0, 1], dtype=complex)

def ketbra(a, b):
    return np.outer(a, np.conjugate(b))   # |a><b|

rho1 = (1/4)*ketbra(ket0, ket0) + (1/4)*ketbra(ket0, ket1) \
     + (1/4)*ketbra(ket1, ket0) + (1/4)*ketbra(ket1, ket1) \
     + (1/2)*ketbra(ket2, ket2)

rho2 = (1/2)*ketbra(ket0, ket0) - (1/2)*ketbra(ket1, ket1) + 1*ketbra(ket2, ket2)

rho3 = 1*ketbra(ket0, ket0) + 1*ketbra(ket1, ket1)

eig1 = np.linalg.eig(rho1)[0]
eig2 = np.linalg.eig(rho2)[0]
eig3 = np.linalg.eig(rho3)[0]

print("A:\n", rho1,"\n",eig1)
print("B:\n", rho2,"\n", eig2)
print("C:\n", rho3,"\n", eig3)
