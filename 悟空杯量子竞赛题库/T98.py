# Bob做X^(M_2)Z^(M_1)操作即可
# 验证代码，采用ChatGPTv5.2thinking编写

import numpy as np
from pyqpanda3.core import (
    QCircuit, QProg, CPUQVM,
    H, CNOT, CZ, U3, S, measure
)

# --------- 工具函数：从 get_counts() 里解析 Bob 的测量，并计算期望值 ----------
def exp_from_counts(counts: dict, bob_bit_pos: int):
    """
    bob_bit_pos: 在 bitstring 中 Bob 的那一位下标（0表示最左边）
    返回：<Z> = P(0)-P(1)（若你在测 X/Y 基前做了相应旋转，则得到 <X>/<Y>）
    """
    total = sum(counts.values())
    z = 0.0
    for bitstr, ct in counts.items():
        b = int(bitstr[bob_bit_pos])
        z += (1 if b == 0 else -1) * ct
    return z / total


# --------- 构造：Teleportation（用“测量延迟”等价的全幺正纠正版本） ----------
def build_teleport_prog(theta, phi, basis="Z", shots=4000):
    """
    qubit 0: |psi>
    qubit 1: Alice 的 Bell 半边
    qubit 2: Bob 的 Bell 半边

    basis: "Z"/"X"/"Y" 表示最后测 Bob 用哪个基（通过末端旋转实现）
    """

    q_psi, q_a, q_b = 0, 1, 2

    cir = QCircuit()

    # 1) 制备要传送的 |psi>：这里用 U3(theta, phi, 0) 在 |0> 上制备
    # |psi> = cos(theta/2)|0> + e^{i*phi} sin(theta/2)|1>
    cir << U3(q_psi, theta, phi, 0.0)

    # 2) 制备 Bell 对：(|00>+|11>)/sqrt(2) 在 (q_a, q_b)
    cir << H(q_a) << CNOT(q_a, q_b)

    # 3) Alice 做 Bell 测量对应的幺正：CNOT(psi->a), H(psi)
    cir << CNOT(q_psi, q_a) << H(q_psi)

    # 4) 纠正：经典规则是 Bob 做 X^{M2} Z^{M1}
    #    用“延迟测量”把它改成测量前的量子控制：
    #    - M2 来自 q_a => CNOT(q_a -> q_b) 等价于“若 M2=1 则对 Bob 做 X”
    #    - M1 来自 q_psi => CZ(q_psi -> q_b) 等价于“若 M1=1 则对 Bob 做 Z”
    cir << CNOT(q_a, q_b) << CZ(q_psi, q_b)

    # 5) 选择 Bob 的测量基（通过末端旋转把 X/Y 基测量转成 Z 测量）
    if basis.upper() == "X":
        cir << H(q_b)
    elif basis.upper() == "Y":
        cir << S(q_b).dagger() << H(q_b)
    elif basis.upper() == "Z":
        pass
    else:
        raise ValueError("basis must be one of: 'X','Y','Z'")

    prog = QProg()
    prog << cir

    # 为了让 bitstring 更好读：把 c2,c1,c0 映射成 q0,q1,q2
    # 假设 get_counts() 返回的是 c2c1c0（常见大端序），则 bitstring = q0 q1 q2
    prog << measure(q_psi, 2) << measure(q_a, 1) << measure(q_b, 0)

    qvm = CPUQVM()
    qvm.run(prog, shots)
    counts = qvm.result().get_counts()

    return prog, counts


if __name__ == "__main__":
    shots = 6000

    # 随便取一个非平凡态，方便验证（你也可以换成别的）
    theta = 1.1
    phi = 0.7

    # 理论 Bloch 分量（对应 |psi> = cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>）
    theo_X = np.sin(theta) * np.cos(phi)
    theo_Y = np.sin(theta) * np.sin(phi)
    theo_Z = np.cos(theta)

    print("Theoretical <X>, <Y>, <Z> =", theo_X, theo_Y, theo_Z)

    # 说明：由于我们把 bitstring 组织成 q0 q1 q2，所以 Bob 是最后一位 => 下标 2
    bob_pos = 2

    for basis in ["X", "Y", "Z"]:
        prog, counts = build_teleport_prog(theta, phi, basis=basis, shots=shots)
        est = exp_from_counts(counts, bob_bit_pos=bob_pos)

        print("\n==============================")
        print(f"Measure Bob in {basis}-basis")
        print("Circuit:")
        print(prog)
        print("Counts (top few):", dict(list(sorted(counts.items(), key=lambda x: -x[1]))[:8]))
        print(f"Estimated <{basis}> =", est)

    print("\nDone.")