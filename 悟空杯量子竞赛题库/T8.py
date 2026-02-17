import numpy as np
from pyqpanda3.core import QCircuit, QProg, CPUQVM, H, U3, measure

def unitary2u3(U: np.ndarray):
    """
    将任意 2x2 酉矩阵 U（允许整体相位）转换为 U3(theta, phi, lam) 参数。
    目标：U3(theta,phi,lam) 与 U 相差一个整体相位（物理等价）。
    """
    U = np.asarray(U, dtype=complex)
    if U.shape != (2, 2):
        raise ValueError("U must be 2x2")

    # 1) 去掉整体相位使 det=1（变成 SU(2)）
    detU = np.linalg.det(U)
    U1 = U * np.exp(-1j * np.angle(detU) / 2)

    # 2) 再去掉整体相位，让 U1[0,0] 变为实数非负（更稳定）
    if abs(U1[0, 0]) > 1e-12:
        U2 = U1 * np.exp(-1j * np.angle(U1[0, 0]))
    else:
        U2 = U1

    # 3) 从标准 U3 形式读出角参数
    # U3 = [[cos(t/2), -e^{iλ} sin(t/2)],
    #       [e^{iφ} sin(t/2),  e^{i(φ+λ)} cos(t/2)]]
    c = np.clip(np.real(U2[0, 0]), -1.0, 1.0)
    theta = 2 * np.arccos(c)
    s = np.sin(theta / 2)

    if abs(s) < 1e-10:
        # theta≈0：只剩对角相位
        phi = 0.0
        lam = np.angle(U2[1, 1])
    else:
        phi = np.angle(U2[1, 0])
        lam = np.angle(-U2[0, 1])

    def wrap(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    return float(wrap(theta)), float(wrap(phi)), float(wrap(lam))

def measure_in_basis_1q(prep, phi_vec, phip_vec, shots=20000, q=0, c=0):
    """
    在任意单比特正交归一测量基 { |phi>, |phi'> } 上测量。
    约定：测量结果 "0" ↔ |phi>, "1" ↔ |phi'>
    """
    phi = np.asarray(phi_vec, dtype=complex).reshape(2)
    phip = np.asarray(phip_vec, dtype=complex).reshape(2)

    # 归一化
    phi = phi / np.linalg.norm(phi)
    phip = phip / np.linalg.norm(phip)

    # 检查正交性（可选但强烈建议）
    if abs(np.vdot(phi, phip)) > 1e-8:
        raise ValueError("(|phi>, |phi'>) is not orthogonal (not a valid projective basis).")

    # U = [|phi>, |phi'>]，在该基上测量 ⇔ 先做 U^\dagger 再计算基测量
    U = np.column_stack([phi, phip])
    U_dag = U.conj().T

    theta, ang_phi, lam = unitary2u3(U_dag)

    prog = QProg()
    prog << prep
    prog << U3(q, theta, ang_phi, lam)
    prog << measure(q, c)

    qvm = CPUQVM()
    qvm.run(prog, shots)
    counts = qvm.result().get_counts()

    n0 = counts.get("0", 0)
    n1 = counts.get("1", 0)
    return counts, (n0 / shots, n1 / shots)

if __name__ == "__main__":
    # 示例：制备 |+>，并用题目那组基测量
    prep = QCircuit()
    prep << H(0)   # |+>

    phi_vec  = np.array([1/2,  np.sqrt(3)/2], dtype=complex)
    phip_vec = np.array([np.sqrt(3)/2, -1/2], dtype=complex)

    counts, (p_phi, p_phip) = measure_in_basis_1q(prep, phi_vec, phip_vec, shots=20000)

    print("counts =", counts)
    print("P(|phi>)  ~", p_phi)
    print("P(|phi'>) ~", p_phip)
