import math
from pyqpanda3.core import QCircuit, QProg, H, X, CR, SWAP, measure, CPUQVM


# ============================================================
# 1. QFT / IQFT 基础模块
# ============================================================

def qft_circuit(qubits, with_swap=True):
    """
    构造 n 比特 QFT 电路
    参数:
        qubits: 量子比特编号列表，例如 [0, 1, 2, 3]
        with_swap: 是否在末尾加入比特反转 SWAP
    返回:
        QCircuit
    """
    cir = QCircuit()
    n = len(qubits)

    # 标准 QFT 结构：
    # 对第 i 个比特，先 H，再接受来自后续比特的受控相位旋转
    for i in range(n):
        target = qubits[i]
        cir << H(target)

        for j in range(i + 1, n):
            control = qubits[j]
            angle = math.pi / (2 ** (j - i))
            cir << CR(control, target, angle)

    # 最后的比特倒序交换
    if with_swap:
        for i in range(n // 2):
            cir << SWAP(qubits[i], qubits[n - 1 - i])

    return cir


def iqft_circuit(qubits, with_swap=True):
    """
    构造 n 比特逆 QFT 电路
    参数:
        qubits: 量子比特编号列表，例如 [0, 1, 2, 3]
        with_swap: 是否包含与 QFT 对应的比特顺序恢复
    返回:
        QCircuit
    """
    cir = QCircuit()
    n = len(qubits)

    # 若 QFT 末尾做过 swap，则 IQFT 开头应先做回来
    if with_swap:
        for i in range(n // 2):
            cir << SWAP(qubits[i], qubits[n - 1 - i])

    # 逆序执行：先逆受控相位，再 H
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
    将 qubits 全部测量到同编号经典寄存器
    例如 qubits=[0,1,2] -> measure(0,0), measure(1,1), measure(2,2)
    """
    if cbits is None:
        cbits = qubits

    for q, c in zip(qubits, cbits):
        prog << measure(q, c)
    return prog


def bit_reverse_indices(n):
    """
    返回 QFT 末尾对应的交换对信息，仅作调试/理解用途
    """
    pairs = []
    for i in range(n // 2):
        pairs.append((i, n - 1 - i))
    return pairs


def demo_qft_on_basis_state():
    """
    示例1：
    制备 |101>，然后施加 QFT，再测量
    用于观察 QFT 后的测量统计分布
    """
    qvm = CPUQVM()
    prog = QProg()

    qubits = [0, 1, 2]

    # 制备 |101>
    prog << X(0)
    prog << X(2)

    # 施加 QFT
    prog << qft_circuit(qubits, with_swap=True)

    # 测量
    append_measure_all(prog, qubits)

    qvm.run(prog, 10000)
    result = qvm.result().get_counts()

    print("=== demo_qft_on_basis_state ===")
    print(prog)
    print("counts =", result)

def demo_qft_then_iqft():
    """
    示例2：
    制备 |101>，先做 QFT，再做 IQFT
    理论上应恢复为原始态 |101>
    """
    qvm = CPUQVM()
    prog = QProg()

    qubits = [0, 1, 2]

    # 制备 |101>
    prog << X(0)
    prog << X(2)

    # 先 QFT
    prog << qft_circuit(qubits, with_swap=True)

    # 再 IQFT
    prog << iqft_circuit(qubits, with_swap=True)

    # 测量
    append_measure_all(prog, qubits)

    qvm.run(prog, 1000)
    result = qvm.result().get_counts()

    print("=== demo_qft_then_iqft ===")
    print(prog)
    print("counts =", result)


if __name__ == "__main__":
    demo_qft_on_basis_state()
    print()
    demo_qft_then_iqft()