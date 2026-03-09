import QFTFunction
from pyqpanda3.core import QProg, X, measure, CPUQVM


def prepare_basis_state_le(prog, qubits, value):
    """
    按小端序方式制备整数value。
    qubits[0] 为最低位。
    """
    for i, q in enumerate(qubits):
        if (value >> i) & 1:
            prog << X(q)
    return prog


if __name__ == "__main__":
    # -----------------------------
    # 参数
    # -----------------------------
    x_value = 3
    y_value = 3
    N_value = 5

    # 因为 N=5，需要 n=3 位寄存器（满足 N < 2^n）
    n = 3

    # -----------------------------
    # 寄存器分配（直接用整数编号）
    # x_qubits, y_qubits, aux0 都按 little-endian 使用
    # -----------------------------
    x_qubits = [0, 1, 2]   # x = 3 -> |011> (little-endian)
    y_qubits = [3, 4, 5]   # y = 4 -> |100> (little-endian)
    aux0 = [6, 7, 8]       # 结果寄存器，初态需为 |000>
    aux1 = 9               # 辅助位，初态需为 |0>

    qvm = CPUQVM()
    prog = QProg()

    # -----------------------------
    # 制备输入态 |x>|y>|0...0>|0>
    # -----------------------------
    prepare_basis_state_le(prog, x_qubits, x_value)
    prepare_basis_state_le(prog, y_qubits, y_value)

    # -----------------------------
    # 调用 QFTVarModAdd
    # 目标：
    # |x>|y>|0...0>|0> -> |x>|y>|(x+y) mod N>|flag>
    # -----------------------------
    prog << QFTFunction.QFTVarModAdd(
        x_qubits,
        y_qubits,
        aux1,
        aux0,
        N_value
    )

    # -----------------------------
    # 这里只测量结果寄存器 aux0 和辅助位 aux1
    # c0, c1, c2 对应 aux0[0], aux0[1], aux0[2]
    # c3 对应 aux1
    # -----------------------------
    prog << measure(aux0[0], 0)
    prog << measure(aux0[1], 1)
    prog << measure(aux0[2], 2)
    prog << measure(aux1, 3)

    qvm.run(prog, 1000)
    result = qvm.result().get_counts()

    print("=== test QFTVarModAdd ===")
    print(f"x = {x_value}, y = {y_value}, N = {N_value}")
    print(f"theory: (x + y) mod N = {(x_value + y_value) % N_value}")
    print("measured qubits:")
    print("  aux0 =", aux0, " (little-endian result register)")
    print("  aux1 =", aux1, " (flag qubit)")
    print(result)