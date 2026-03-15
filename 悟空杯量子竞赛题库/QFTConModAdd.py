import math
from pyqpanda3.core import QCircuit, QProg, H, X, CR, CNOT, SWAP, measure, CPUQVM, U1
from QFTConAdd import append_measure_all, QFTConAdd, getAngles, qft_circuit, iqft_circuit

def prepare_basis_state_le(prog, qubits, value):
    """
    按 little-endian 方式制备整数 value。
    即 qubits[0] 对应最低位，qubits[1] 对应次低位，依此类推。
    """
    for i, q in enumerate(qubits):
        if (value >> i) & 1:
            prog << X(q)
    return prog

def CtrlQFTConAdd(qubits, ctrl, a):
    """
    基于 QFT 的受控常数加法器。

    参数:
        qubits: [MSB, ..., LSB]
        ctrl:   控制位
        a:      要加的常数（模 2^n）

    实现:
        当 ctrl=1 时，|x> -> |x + a (mod 2^n)>
        当 ctrl=0 时，不变。
    """
    cir = QCircuit()
    angles = getAngles(a, len(qubits))

    cir << qft_circuit(qubits, with_swap=False)
    for q, theta in zip(qubits, angles):
        cir << CR(ctrl, q, theta)
    cir << iqft_circuit(qubits, with_swap=False)

    return cir

def ConModAdd(x_qubits, work_msb, aux, a, N):
    """
    基于 QFT 的常数模加法器。

    实现目标:
        |x> -> |(x + a) mod N>

    参数:
        x_qubits : 数据寄存器（little-endian，x_qubits[0] 为最低位）
        work_msb : 工作寄存器最高位/借位位（初态必须为 |0>）
        aux      : 辅助控制位（初态必须为 |0>）
        a        : 常数加数，要求 0 <= a < N
        N        : 模数

    说明:
        这里把完整工作寄存器写成
            full_reg = [work_msb] + x_qubits[::-1]
        也就是按 [MSB, ..., LSB] 的顺序传给 QFT 加法器。

    线路思想与题图一致：
        1) 先计算 x + a - N
        2) 取最高位到 aux，判断是否为负
        3) 若 aux = 1，则加回 N
        4) 通过减 a / 判符号 / 加 a，将 aux 反计算回 |0>
    """
    cir = QCircuit()
    n = len(x_qubits)

    if not (0 <= a < N):
        raise ValueError("要求满足 0 <= a < N")
    if N >= (1 << n):
        raise ValueError("要求模数 N 小于 2^n，x_qubits 位数不足")

    # 算术顺序: [MSB, ..., LSB]
    full_reg = [work_msb] + list(reversed(x_qubits))

    # Step 1: 计算 x + a - N
    cir << QFTConAdd(full_reg, a - N)

    # Step 2: 提取最高位到 aux
    cir << CNOT(work_msb, aux)

    # Step 3: 如果 aux = 1，说明 x + a < N，需要加回 N
    cir << CtrlQFTConAdd(full_reg, aux, N)

    # Step 4: 反计算 aux，使其回到 |0>
    # 此时 full_reg 已是 (x + a) mod N
    # 先减 a，再利用最高位与 aux 的关系把 aux 清零，最后加回 a
    cir << QFTConAdd(full_reg, -a)
    cir << X(aux)
    cir << CNOT(work_msb, aux)
    cir << QFTConAdd(full_reg, a)

    return cir

# 示例

def demo_con_mod_add():
    """
    位序说明:
        x_qubits 采用 little-endian 存数：
            x_qubits[0] 是最低位
            x_qubits[1] 是次低位
            x_qubits[2] 是最高位
    """
    qvm = CPUQVM()
    prog = QProg()

    x_qubits = [0, 1, 2]   # little-endian: q0 为最低位
    work_msb = 3           # 工作寄存器最高位/借位位
    aux = 4                # 辅助位

    x_value = 4
    a_value = 4
    N_value = 5

    # 制备 |x>
    prepare_basis_state_le(prog, x_qubits, x_value)

    # 施加常数模加法器
    prog << ConModAdd(x_qubits, work_msb, aux, a_value, N_value)

    # 测量全部相关比特
    all_qubits = x_qubits + [work_msb, aux]
    append_measure_all(prog, all_qubits)

    qvm.run(prog, 1000)
    result = qvm.result().get_counts()

    print("=== demo_con_mod_add ===")
    print(f"x = {x_value}, a = {a_value}, N = {N_value}")
    print(f"theory: (x + a) mod N = {(x_value + a_value) % N_value}")
    print("bit order:")
    print("  x_qubits = [q0, q1, q2] are little-endian (q0 is LSB)")
    print("  measured qubits = [q0, q1, q2, work_msb, aux]")
    print(prog)
    print("counts =", result)


if __name__ == "__main__":
    demo_con_mod_add()
