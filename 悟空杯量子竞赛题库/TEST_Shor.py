import math
import numpy as np
import QFTFunction
from Shor import Shor, CFD

from pyqpanda3.core import QProg, measure, CPUQVM

# 模板
"""
        Shor函数与CFD函数套用模板
    # 1. 设参数
    N = ...
    a = ...
    phase_qubits = ...
    acc_qubits = ...
    tmp_qubits = ...
    work_msb = ...
    aux = ...
    
    # 2. 搭电路
    cir = Shor(phase_qubits, acc_qubits, tmp_qubits, work_msb, aux, a, N)
    
    # 3. 加测量，只测第一寄存器
    prog << cir
    for i, q in enumerate(phase_qubits):
        prog << measure(q, i)
    
    # 4. 运行得到 counts
    counts = qvm.result().get_counts()
    
    # 5. 连分数恢复周期
    r = CFD(counts, a, N, t=len(phase_qubits), bit_order="msb", top_k=8)
"""

def main():
    N = 15
    a = 2

    # 对 N=15，结果寄存器需要至少 4 位，因为 15 < 2^4
    n = 4

    # 第一寄存器位数 t
    # Shor 中常取 Q = 2^t >= N^2
    # 这里 N^2 = 225，因此取 t = 8，Q = 256
    t = 8

    # phase_qubits: little-endian
    #   phase_qubits[0] 为最低位，对应 2^0
    phase_qubits = list(range(0, t))                # [0,1,2,3,4,5,6,7]

    # acc_qubits: 第二寄存器，little-endian
    acc_qubits = list(range(t, t + n))              # [8,9,10,11]

    # tmp_qubits: 模乘辅助寄存器，little-endian
    tmp_qubits = list(range(t + n, t + 2 * n))      # [12,13,14,15]

    # work_msb, aux: 额外辅助位
    work_msb = t + 2 * n                            # 16
    aux = t + 2 * n + 1                             # 17

    cir = Shor(
        phase_qubits=phase_qubits,
        acc_qubits=acc_qubits,
        tmp_qubits=tmp_qubits,
        work_msb=work_msb,
        aux=aux,
        a=a,
        N=N
    )

    prog = QProg()
    prog << cir

    # phase_qubits 是 little-endian: [LSB, ..., MSB]
    # 这里把 q_i 测到 c_i
    # 若 counts 的字符串按高经典位 -> 低经典位显示，
    # 则打印出来的 bitstring 正好是 MSB -> LSB，可直接给 CFD(bit_order="msb")
    for i, q in enumerate(phase_qubits):
        prog << measure(q, i)

    qvm = CPUQVM()
    shots = 4096
    qvm.run(prog, shots)
    counts = qvm.result().get_counts()

    print("N =", N, ", a =", a)
    print("Theoretical order r = 4")
    print("Shots =", shots)
    print("Raw counts =", counts)

    # 显示前若干个主峰，便于观察
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    print("\nTop measurement peaks:")
    for bits, cnt in sorted_counts[:8]:
        print(bits, ":", cnt)

    r = CFD(
        meas_result=counts,
        a=a,
        N=N,
        t=t,
        bit_order="msb",
        top_k=8
    )

    print("\nRecovered period r =", r)

    if r is not None:
        print("Check: a^r mod N =", pow(a, r, N))
        if r % 2 == 0:
            x = pow(a, r // 2, N)
            print("a^(r/2) mod N =", x)
            print("gcd(a^(r/2)-1, N) =", math.gcd(x - 1, N))
            print("gcd(a^(r/2)+1, N) =", math.gcd(x + 1, N))
    else:
        print("CFD 未恢复出周期，可重跑一次或适当增大 shots。")


if __name__ == "__main__":
    main()