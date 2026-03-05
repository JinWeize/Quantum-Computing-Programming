import numpy as np
from pyqpanda3.quantum_info.quantum_info import StateVector, DensityMatrix
from pyqpanda3.core import H, X, CNOT, QCircuit, QProg, CZ, RX, CP,DensityMatrixSimulator

psi = [0.5 + 0j, (3**0.5)/2 + 0j]      # |psi>
sv = StateVector(psi)                  # 支持用 list[complex] 构造 StateVector
rho = DensityMatrix(sv)                # 支持用 StateVector 构造 DensityMatrix

print("is_valid:", rho.is_valid())  # 判断矩阵数据是否为合法的密度矩阵
print("purity:", rho.purity())  # 计算密度矩阵的纯度
print("rho numpy:\n", rho.ndarray())    # 导出为numpy.ndarray
print("rho[0,1] =", rho.at(0,1))    # 读取第（0， 1）个矩阵元素

print("*******************************************************************")

rho_np = np.array([[0.5+0j, 0j],
                   [0j, 0.5+0j]], dtype=np.complex128)

rho = DensityMatrix(rho_np)            # 支持 numpy 2D 数组构造
print("is_valid:", rho.is_valid())
print("purity:", rho.purity())         # 这里应接近 0.5（单比特最大混态）

print("*******************************************************************")

rho0 = DensityMatrix(1)       # 1 qubit, 默认为|0><0|

cir = QCircuit(1)
cir << H(0)

rho1 = rho0.evolve(cir)       # 不改 rho0，返回新的演化后的 rho1

print("rho0:\n", rho0.ndarray())
print("rho1:\n", rho1.ndarray())
print("rho1 purity:", rho1.purity())  # 仍为 1（|+><+|）

print("*******************************************************************")

circuit = QCircuit(2)
circuit << H(0) << H(1) << RX(0, 1.5) << CZ(0, 1) << CP(1, 0, 0.22)

prog = QProg()
prog << circuit

machine = DensityMatrixSimulator()
machine.run(prog)
rho_np = machine.density_matrix()             # 全体系密度矩阵
rho0_np = machine.reduced_density_matrix([0]) # 子系统约化密度矩阵

# rho = DensityMatrix(rho_np)                   #再封装成 DensityMatrix便于purity/is_valid等，此两条代码bug
# print("purity:", rho.purity())