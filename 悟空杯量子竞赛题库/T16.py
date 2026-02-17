import numpy as np

# 基本门
I = np.array([[1, 0],
              [0, 1]], dtype=complex)

Y = np.array([[0, -1j],
              [1j, 0]], dtype=complex)

H = (1/np.sqrt(2)) * np.array([[1,  1],
                               [1, -1]], dtype=complex)

# 初态 |0><0|
rho0 = np.array([[1, 0],
                 [0, 0]], dtype=complex)

# 概率
pI, pY, pH = 2/5, 1/5, 2/5

rho = pI * (I @ rho0 @ I.conj().T) \
    + pY * (Y @ rho0 @ Y.conj().T) \
    + pH * (H @ rho0 @ H.conj().T)

print("rho =\n", rho)
print("rho * 5 =\n", 5*rho)
