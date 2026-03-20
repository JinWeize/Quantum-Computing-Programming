import math
import numpy as np

print("=============第一问==============")
# 在|0>,|1>下的a，b
ket_a = np.array([1/math.sqrt(6) + 1/math.sqrt(3), 1/math.sqrt(6)-1/math.sqrt(3)])
ket_a = ket_a.reshape(2, 1)
ket_b = np.array([math.sqrt(2)/3 - math.sqrt(5)/(3 * math.sqrt(2)), math.sqrt(2)/3 + math.sqrt(5)/(3*math.sqrt(2))])
ket_b = ket_b.reshape(2, 1)
H = 1/math.sqrt(2) * np.array([[1, 1],[1, -1]])

rho_01 = 3 / 4 * (ket_a @ ket_a.conj().T) + 1 / 4 * (ket_b @ ket_b.conj().T)
rho_np = H.conj().T @ rho_01 @ H

print("|0>|1>测量基下的密度矩阵\n",rho_01)
print("\n|+||-|测量基下的密度矩阵\n",rho_np)

print("=============第二问==============")
# 在|+>|->基下进行测量
p = np.array([1/math.sqrt(2), 1/math.sqrt(2)])
ket_1 = np.array([1, 0])
n = np.array([1/math.sqrt(2), -1/math.sqrt(2)])
ket_0 = np. array([0, 1])
p = p.reshape(2, 1)
n = n.reshape(2, 1)
ket_1 = ket_1.reshape(2, 1)
ket_0 = ket_0.reshape(2, 1)

prob_p = np.trace(ket_1 @ ket_1.conj().T @ rho_np)
prob_n = np. trace(ket_0 @ ket_0.conj().T @ rho_np)

print("\n|+||-|测量基下|+>概率\n",prob_p)
print("\n|+||-|测量基下|->概率\n",prob_n)

print("=============第三问==============")

prob_1 = np.trace(ket_1 @ ket_1.conj().T @ rho_01)
prob_0 = np. trace(ket_0 @ ket_0.conj().T @ rho_01)

print("\n|0||1|测量基下|1>概率\n",prob_1)
print("\n|0||1|测量基下|0>概率\n",prob_0)

print("=============第四问==============")
X = np.array([[0, 1], [1, 0]])

E_X = np.trace(rho_01 @ X)
print(E_X)