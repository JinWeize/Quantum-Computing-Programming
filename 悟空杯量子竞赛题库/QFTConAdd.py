import math
from pyqpanda3.core import QCircuit, QProg, H, X, CR, SWAP, RZ, measure, CPUQVM, U1


# ============================================================
# 1. QFT / IQFT 基础模块
# ============================================================

def qft_circuit(qubits, with_swap=True):
    """
    构造 n 比特 QFT 电路。

    参数:
        qubits: 量子比特编号列表，例如 [0, 1, 2]
        with_swap: 是否在末尾加入比特反转 SWAP

    返回:
        QCircuit
    """
    cir = QCircuit()
    n = len(qubits)

    for i in range(n):
        target = qubits[i]
        cir << H(target)
        for j in range(i + 1, n):
            control = qubits[j]
            angle = math.pi / (2 ** (j - i))
            cir << CR(control, target, angle)

    if with_swap:
        for i in range(n // 2):
            cir << SWAP(qubits[i], qubits[n - 1 - i])

    return cir


def iqft_circuit(qubits, with_swap=True):
    """
    构造 n 比特 IQFT 电路。

    参数:
        qubits: 量子比特编号列表，例如 [0, 1, 2]
        with_swap: 是否包含与 QFT 对应的比特顺序恢复

    返回:
        QCircuit
    """
    cir = QCircuit()
    n = len(qubits)

    if with_swap:
        for i in range(n // 2):
            cir << SWAP(qubits[i], qubits[n - 1 - i])

    for i in range(n - 1, -1, -1):
        target = qubits[i]
        for j in range(n - 1, i, -1):
            control = qubits[j]
            angle = -math.pi / (2 ** (j - i))
            cir << CR(control, target, angle)
        cir << H(target)

    return cir


# ============================================================
# 2. 常用辅助函数
# ============================================================

def append_measure_all(prog, qubits, cbits=None):
    """
    将 qubits 全部测量到对应经典寄存器。
    """
    if cbits is None:
        cbits = qubits

    for q, c in zip(qubits, cbits):
        prog << measure(q, c)
    return prog


# ============================================================
# 3. 基于 QFT 的量子常数加法器
# ============================================================

def getAngles(a, n):
    mod = 1 << n
    a = a % mod

    # 低位在前：bits[j] 对应 2^j 位
    bits = [(a >> j) & 1 for j in range(n)]

    angles = []
    for i in range(n):
        theta = 0.0
        for j in range(n - i):
            if bits[j]:
                theta += 2 * math.pi / (2 ** (n - i - j))
        angles.append(theta)

    return angles



def QFTConAdd(qubits, a, with_swap=True):
    cir = QCircuit()
    n = len(qubits)
    angles = getAngles(a, n)

    cir << qft_circuit(qubits, with_swap=False)

    for q, theta in zip(qubits, angles):
        cir << U1(q, theta)

    cir << iqft_circuit(qubits, with_swap=False)
    return cir


# ============================================================
# 4. 使用示例
# ============================================================

def demo_qft_const_add():
    qvm = CPUQVM()
    prog = QProg()

    qubits = [0, 1, 2, 4]
    x_value = 3
    a_value = 5

    # 制备 |011>（按当前文件的位编号约定，X(0), X(1) 置 1）
    prog << X(0)
    prog << X(1)

    # 施加基于 QFT 的常数加法器：|x> -> |x + a mod 2^n>
    prog << QFTConAdd(qubits[::-1], a_value, with_swap=False)

    # 测量
    append_measure_all(prog, qubits)

    qvm.run(prog, 1000)
    result = qvm.result().get_counts()

    print("=== demo_qft_const_add ===")
    print(f"n = {len(qubits)}, x = {x_value}, a = {a_value}")
    print(f"angles = {getAngles(a_value, len(qubits))}")
    print(prog)
    print("counts =", result)


if __name__ == "__main__":
    demo_qft_const_add()
