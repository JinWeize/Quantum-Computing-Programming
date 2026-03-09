import QFTFunction
from pyqpanda3.core import QProg, X, measure, CPUQVM


def prepare_basis_state_le(prog, qubits, value):
    """
    按 little-endian 方式制备整数 value。
    qubits[0] 为最低位。
    """
    for i, q in enumerate(qubits):
        if (value >> i) & 1:
            prog << X(q)
    return prog


def append_measure_all(prog, qubits):
    """
    按给定顺序把 qubits 全部测量。
    这里默认量子位和经典位编号相同。
    """
    for q in qubits:
        prog << measure(q, q)
    return prog


if __name__ == "__main__":
    qvm = CPUQVM()
    prog = QProg()

    # 参数设置
    a_value = 2
    N_value = 5
    x_value = 3

    # 寄存器分配（直接用整数编号）
    # exp_qubits: 指数寄存器 |x>
    # acc_qubits: 累乘/结果寄存器，初态必须为 |1>
    # tmp_qubits: 临时寄存器，初态必须为 |0>
    exp_qubits = [0, 1]        # little-endian，x=3 即 |11>
    acc_qubits = [2, 3, 4]     # little-endian，初态置为 |1>
    tmp_qubits = [5, 6, 7]     # little-endian，初态 |0>
    work_msb = 8               # 辅助工作位，初态 |0>
    aux = 9                    # 辅助位，初态 |0>

    # 初态制备
    # |x>|1>|0>
    prepare_basis_state_le(prog, exp_qubits, x_value)
    prepare_basis_state_le(prog, acc_qubits, 1)

    # 调用 QFT 常数模幂运算器
    prog << QFTFunction.QFTConModExp(
        exp_qubits,
        acc_qubits,
        tmp_qubits,
        work_msb,
        aux,
        a_value,
        N_value
    )

    # 测量
    all_qubits = exp_qubits + acc_qubits + tmp_qubits + [work_msb, aux]
    append_measure_all(prog, all_qubits)

    qvm.run(prog, 1000)
    result = qvm.result().get_counts()

    print("=== test QFTConModExp ===")
    print(f"a = {a_value}, N = {N_value}, x = {x_value}")
    print(f"theory: a^x mod N = {pow(a_value, x_value, N_value)}")
    print("register order in measurement:")
    print("  exp_qubits =", exp_qubits, " (little-endian)")
    print("  acc_qubits =", acc_qubits, " (little-endian, expected value = 3)")
    print("  tmp_qubits =", tmp_qubits, " (little-endian, expected value = 0)")
    print("  work_msb =", work_msb, ", aux =", aux)
    print(result)