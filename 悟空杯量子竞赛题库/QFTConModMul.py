
import math
from pyqpanda3.core import (
    QCircuit, QProg, H, X, CR, CNOT, SWAP, measure,
    CPUQVM, U1, TOFFOLI
)
import QFTFunction

def append_measure_all(prog, qubits, cbits=None):
    """按给定顺序把 qubits 全部测量到 cbits。"""
    if cbits is None:
        cbits = qubits
    for q, c in zip(qubits, cbits):
        prog << measure(q, c)
    return prog


def prepare_basis_state_le(prog, qubits, value):
    """
    按 little-endian 方式制备整数 value。
    即 qubits[0] 对应最低位，qubits[1] 对应次低位，依此类推。
    """
    for i, q in enumerate(qubits):
        if (value >> i) & 1:
            prog << X(q)
    return prog


def egcd(a, b):
    """扩展欧几里得算法。"""
    if b == 0:
        return a, 1, 0
    g, x1, y1 = egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


def modinv(a, N):
    """求 a 在模 N 下的逆元。要求 gcd(a, N) = 1。"""
    g, x, _ = egcd(a, N)
    if g != 1:
        raise ValueError("a 与 N 不互素，模逆元不存在，无法构造可逆的常数模乘法器")
    return x % N


def qft_circuit(qubits, with_swap=False):
    """
    构造 QFT 电路。
    这里默认 with_swap=False，便于与角度数组和算术位序保持一致。

    参数中的 qubits 采用算术顺序 [MSB, ..., LSB]。
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


def iqft_circuit(qubits, with_swap=False):
    """
    构造逆 QFT 电路。
    参数中的 qubits 采用算术顺序 [MSB, ..., LSB]。
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


def getAngles(a, n):
    """
    计算基于 QFT 的常数加法器所需相位数组。

    这里适配 qft_circuit(..., with_swap=False) 且 qubits 采用 [MSB,...,LSB] 的约定。
    返回的 angles 与传入 qft_circuit 的 qubits 顺序一一对应。
    """
    a = a % (1 << n)
    bits = [(a >> k) & 1 for k in range(n)]  # bits[0] 为最低位

    angles = []
    # 对于 [MSB,...,LSB] 顺序中的第 i 个 qubit，其对应 little-endian 位编号为 n-1-i
    for i in range(n):
        le_idx = n - 1 - i
        theta = 0.0
        for k in range(le_idx + 1):
            if bits[k]:
                theta += 2 * math.pi / (2 ** (le_idx - k + 1))
        angles.append(theta)

    return angles


def QFTConAdd(qubits, a):
    """
    基于 QFT 的常数加法器。

    参数:
        qubits: 算术顺序 [MSB, ..., LSB]
        a:     要加的常数（模 2^n）
    """
    cir = QCircuit()
    angles = getAngles(a, len(qubits))

    cir << qft_circuit(qubits, with_swap=False)
    for q, theta in zip(qubits, angles):
        cir << U1(q, theta)
    cir << iqft_circuit(qubits, with_swap=False)

    return cir


def CtrlQFTConAdd(qubits, ctrl, a):
    """
    基于 QFT 的受控常数加法器。

    当 ctrl=1 时：|x> -> |x + a (mod 2^n)>
    当 ctrl=0 时：不变

    参数:
        qubits: 算术顺序 [MSB, ..., LSB]
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
        x_qubits : little-endian，x_qubits[0] 为最低位
        work_msb : 工作寄存器最高位/借位位（初态 |0>）
        aux      : 辅助位（初态 |0>）
        a        : 常数加数，要求 0 <= a < N
        N        : 模数
    """
    cir = QCircuit()
    n = len(x_qubits)

    if not (0 <= a < N):
        raise ValueError("要求满足 0 <= a < N")
    if N >= (1 << n):
        raise ValueError("要求模数 N 小于 2^n，x_qubits 位数不足")

    full_reg = [work_msb] + list(reversed(x_qubits))

    # 1) x + a - N
    cir << QFTConAdd(full_reg, a - N)
    # 2) 取最高位到 aux
    cir << CNOT(work_msb, aux)
    # 3) 若 aux=1，则加回 N
    cir << CtrlQFTConAdd(full_reg, aux, N)
    # 4) 反计算 aux
    cir << QFTConAdd(full_reg, -a)
    cir << X(aux)
    cir << CNOT(work_msb, aux)
    cir << QFTConAdd(full_reg, a)

    return cir


def CtrlConModAdd(x_qubits, work_msb, aux, ctrl, a, N):
    """
    受控常数模加法器。

    当 ctrl=1 时，|x> -> |(x + a) mod N>
    当 ctrl=0 时，|x> 不变
    """
    cir = QCircuit()
    n = len(x_qubits)

    if not (0 <= a < N):
        raise ValueError("要求满足 0 <= a < N")
    if N >= (1 << n):
        raise ValueError("要求模数 N 小于 2^n，x_qubits 位数不足")

    full_reg = [work_msb] + list(reversed(x_qubits))

    # 1) 当 ctrl=1 时计算 x + a - N
    cir << CtrlQFTConAdd(full_reg, ctrl, a - N)
    # 2) 用 aux 记录 ctrl AND work_msb
    cir << TOFFOLI(ctrl, work_msb, aux)
    # 3) 若 aux=1，则加回 N
    cir << CtrlQFTConAdd(full_reg, aux, N)
    # 4) 反计算 aux
    cir << CtrlQFTConAdd(full_reg, ctrl, -a)
    cir << CNOT(ctrl, aux)
    cir << TOFFOLI(ctrl, work_msb, aux)
    cir << CtrlQFTConAdd(full_reg, ctrl, a)

    return cir


def AMULN(x_qubits, y_qubits, work_msb, aux, a, N):
    """
    基于 QFT 的常数模加乘运算器。

    实现目标:
        |x>|y> -> |x>|(y + a*x) mod N>
    """
    cir = QCircuit()
    n = len(x_qubits)

    if len(y_qubits) != n:
        raise ValueError("这里要求 x_qubits 与 y_qubits 位数一致")
    if not (0 <= a < N):
        raise ValueError("要求满足 0 <= a < N")
    if N >= (1 << n):
        raise ValueError("要求模数 N 小于 2^n，寄存器位数不足")

    for j, ctrl in enumerate(x_qubits):
        add_const = (a * (1 << j)) % N
        cir << CtrlConModAdd(y_qubits, work_msb, aux, ctrl, add_const, N)

    return cir


def ConModMul(x_qubits, y_qubits, work_msb, aux, a, N):
    """
    基于 QFT 的常数模乘法器。

    实现目标:
        在 y 寄存器初态为 |0> 时，实现
            |x>|0> -> |a*x mod N>|0>

    参数:
        x_qubits : 输入/输出寄存器，little-endian
        y_qubits : 辅助结果寄存器，little-endian，要求初态为 |0>
        work_msb : 作用在加法目标寄存器上的工作最高位/借位位（初态 |0>）
        aux      : 辅助位（初态 |0>）
        a        : 常数，要求 0 <= a < N 且 gcd(a, N)=1
        N        : 模数
    """
    cir = QCircuit()
    n = len(x_qubits)

    if len(y_qubits) != n:
        raise ValueError("这里要求 x_qubits 与 y_qubits 位数一致")
    if not (0 <= a < N):
        raise ValueError("要求满足 0 <= a < N")
    if N >= (1 << n):
        raise ValueError("要求模数 N 小于 2^n，寄存器位数不足")

    a_inv = modinv(a, N)

    # 1) |x>|0> -> |x>|a*x mod N>
    cir << AMULN(x_qubits, y_qubits, work_msb, aux, a, N)

    # 2) 交换两个寄存器
    for qx, qy in zip(x_qubits, y_qubits):
        cir << SWAP(qx, qy)

    # 3) 清空第二寄存器：y <- y - a_inv * x (mod N)
    cir << AMULN(x_qubits, y_qubits, work_msb, aux, (-a_inv) % N, N)

    return cir


def demo_con_mod_mul():
    """
    示例：演示 ConModMul 的使用方式。

    位序约定:
        x_qubits, y_qubits 都采用 little-endian：
            q[0] 是最低位，q[1] 是次低位，依此类推。

    本例目标变换:
        |x>|0> -> |a*x mod N>|0>
    """
    qvm = CPUQVM()
    prog = QProg()

    x_qubits = [0, 1, 2]
    y_qubits = [3, 4, 5]
    work_msb = 6
    aux = 7

    x_value = 3
    a_value = 2
    N_value = 5
    y_value = 0

    prepare_basis_state_le(prog, x_qubits, x_value)
    prepare_basis_state_le(prog, y_qubits, y_value)

    prog << QFTFunction.ConMULTN(x_qubits, y_qubits, work_msb, aux, a_value, N_value)

    all_qubits = x_qubits + y_qubits + [work_msb, aux]
    append_measure_all(prog, all_qubits)

    qvm.run(prog, 1000)
    result = qvm.result().get_counts()

    theory = (a_value * x_value) % N_value

    print("=== demo_con_mod_mul ===")
    print(f"x = {x_value}, a = {a_value}, N = {N_value}")
    print(f"theory: (a*x) mod N = {theory}")
    print("bit order:")
    print("  x_qubits = [q0, q1, q2] are little-endian")
    print("  y_qubits = [q3, q4, q5] are little-endian")
    print("  measured qubits = [x_qubits, y_qubits, work_msb, aux]")
    print(prog)
    print("counts =", result)


if __name__ == "__main__":
    demo_con_mod_mul()
