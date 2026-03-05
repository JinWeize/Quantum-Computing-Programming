import numpy as np
from pyqpanda3.core import QCircuit, QProg, H, CNOT, DensityMatrixSimulator
from pyqpanda3.quantum_info.quantum_info import DensityMatrix

def to_dm_safe(mat_like):
    """把 simulator 返回的数据稳健地转成合法的密度矩阵 numpy 方阵（尽量修正数值误差）"""
    rho = np.asarray(mat_like, dtype=np.complex128)
    if rho.ndim == 1:
        dim = int(round(np.sqrt(rho.size)))
        rho = rho.reshape((dim, dim))
    # 强制厄米
    rho = (rho + rho.conj().T) / 2
    # 迹归一化
    tr = np.trace(rho)
    if abs(tr) < 1e-15:
        raise RuntimeError("Trace ~ 0：可能是 simulator 未正确生成密度矩阵")
    rho = rho / tr
    # 轻微负特征值截断（防止数值误差）
    w, v = np.linalg.eigh(rho)
    w[w < 0] = 0.0
    rho = (v * w) @ v.conj().T
    rho = rho / np.trace(rho)
    return rho

# 1) 制备 Bell 态电路：|00> -> (|00>+|11>)/sqrt2
cir = QCircuit(2)
cir << H(0) << CNOT(0, 1)

prog = QProg()
prog << cir

# 2) 用 DensityMatrixSimulator 跑
machine = DensityMatrixSimulator()
machine.run(prog)

# 3) 取全体系密度矩阵 & 单比特约化密度矩阵
rho_full_raw = machine.density_matrix()
rho0_raw = machine.reduced_density_matrix([0])   # 只保留第0个比特

rho_full = to_dm_safe(rho_full_raw)
rho0 = to_dm_safe(rho0_raw)

dm_full = DensityMatrix(rho_full)
dm0 = DensityMatrix(rho0)

print("初始纯度=", dm_full.purity())   # 期望 ~ 1
print("约化后的纯度=", dm0.purity())     # 期望 ~ 0.5
print("约化矩阵:\n", dm0.ndarray())   # 期望接近 [[0.5,0],[0,0.5]]