import math
import numpy as np
from pyqpanda3.core import CPUQVM, QProg, QCircuit, H, X, CZ


def add_grover_coin(circ: QCircuit, coin_qubits):
    """
    2比特 Grover coin:
        G = H^{\otimes 2} X^{\otimes 2} CZ X^{\otimes 2} H^{\otimes 2}
    """
    c0, c1 = coin_qubits

    circ << H(c0) << H(c1)
    circ << X(c0) << X(c1)
    circ << CZ(c0, c1)
    circ << X(c0) << X(c1)
    circ << H(c0) << H(c1)


def add_shift_operator(circ: QCircuit, pos_qubits, coin_qubits):
    """
    仿照书上的 Q4 移位算符：
    硬币态 |00>, |01>, |10>, |11> 分别控制翻转位置比特 q0, q1, q2, q3

    这里采用“先X把0控制变成1控制”的写法，与书上代码一致：
        |00> -> flip q0
        |01> -> flip q1
        |10> -> flip q2
        |11> -> flip q3
    """
    q0, q1, q2, q3 = pos_qubits
    c0, c1 = coin_qubits

    # coin = |00> -> flip q0
    circ << X(c0) << X(c1)
    circ << X(q0).control(c0).control(c1)

    # coin = |01> -> flip q1
    circ << X(c0)
    circ << X(q1).control(c0).control(c1)

    # coin = |10> -> flip q2
    circ << X(c1)
    circ << X(q2).control(c0).control(c1)

    # coin = |11> -> flip q3
    circ << X(q3).control(c0).control(c1)


def build_quantum_walk_circuit(n_pos=4):
    """
    构造离散时间量子行走线路
    """
    n_coin = math.ceil(math.log2(n_pos))   # 对 Q4 来说 = 2
    n_qubits = n_pos + n_coin

    pos_qubits = list(range(n_pos))                 # [0,1,2,3]
    coin_qubits = list(range(n_pos, n_qubits))     # [4,5]

    circ = QCircuit()

    # 初态：位置和硬币都制备成均匀叠加态
    for q in pos_qubits + coin_qubits:
        circ << H(q)

    # 按书中写法，步数取 sqrt(N)，其中 N = 2^n_pos
    N = 2 ** n_pos
    n_steps = int(math.sqrt(N))   # Q4 -> 4 步

    for _ in range(n_steps):
        add_grover_coin(circ, coin_qubits)
        add_shift_operator(circ, pos_qubits, coin_qubits)

    return circ, pos_qubits, coin_qubits, n_qubits


def get_position_probabilities(statevector, pos_qubits, n_qubits):
    """
    从总态矢量中把 coin 寄存器迹掉，得到 position 寄存器的概率分布

    约定：
    - 用索引 idx 的二进制表示基态
    - 假设 q0 对应最低位（little-endian）
    - 输出时打印为 |q3 q2 q1 q0> 的常见阅读顺序
    """
    prob_dict = {}

    for idx, amp in enumerate(statevector):
        p = abs(amp) ** 2
        if p < 1e-15:
            continue

        # little-endian: bits_le[k] 是 qk 的比特值
        bits_le = format(idx, f"0{n_qubits}b")[::-1]

        # 打印成高位在左：q3 q2 q1 q0
        pos_bits = "".join(bits_le[q] for q in reversed(pos_qubits))

        prob_dict[pos_bits] = prob_dict.get(pos_bits, 0.0) + p

    # 数值清理
    for k in prob_dict:
        prob_dict[k] = float(np.real_if_close(prob_dict[k]))

    return dict(sorted(prob_dict.items(), key=lambda x: x[0]))


def main():
    qvm = CPUQVM()

    circ, pos_qubits, coin_qubits, n_qubits = build_quantum_walk_circuit(n_pos=4)

    prog = QProg()
    prog << circ

    # 运行并取态矢量
    qvm.run(prog, 1)
    statevector = qvm.result().get_state_vector()

    # 统计位置寄存器概率
    pos_prob = get_position_probabilities(statevector, pos_qubits, n_qubits)

    print("=== Quantum Walk Circuit ===")
    print(prog)

    print("\n=== Position Register Probability Distribution ===")
    for state, prob in pos_prob.items():
        print(f"{state}: {prob:.6f}")

    print(f"\nProbability sum = {sum(pos_prob.values()):.6f}")


if __name__ == "__main__":
    main()