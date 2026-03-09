import math
import QFTFunction

from pyqpanda3.core import QProg, QCircuit, U1, X, measure, CPUQVM


def main():
    phase_qubits = [0, 1, 2]   # 3位相位寄存器
    sys_qubits = [3]           # 1位系统寄存器

    unitary_cir = QCircuit()
    unitary_cir << U1(sys_qubits[0], math.pi / 4)

    state_prep = QCircuit()
    state_prep << X(sys_qubits[0])

    qpe_cir = QFTFunction.QFTPhaseEstimation(
        phase_qubits,
        sys_qubits,
        unitary_cir,
        state_prep
    )

    prog = QProg()
    prog << qpe_cir

    # =========================
    # 5. 测量相位寄存器
    # =========================
    # 这里特意把：
    #   q0(MSB) -> c2
    #   q1      -> c1
    #   q2(LSB) -> c0
    # 这样打印出来的 bitstring 更容易按 MSB->LSB 读成 "001"
    prog << measure(phase_qubits[0], 2)
    prog << measure(phase_qubits[1], 1)
    prog << measure(phase_qubits[2], 0)

    # =========================
    # 6. 运行
    # =========================
    qvm = CPUQVM()
    shots = 1000
    qvm.run(prog, shots)
    counts = qvm.result().get_counts()

    print("Theoretical phase phi = 1/8 = 0.001_2")
    print("Measurement counts:", counts)

    # 取出现次数最多的结果
    best_bits = max(counts, key=counts.get)
    est_phi = int(best_bits, 2) / (2 ** len(phase_qubits))

    print("Most likely phase bits:", best_bits)
    print("Estimated phi =", est_phi)
    print(prog)

if __name__ == "__main__":
    main()