# 量子乘法器（基于 CDKM/Cuccaro ripple-carry 加法器）
# 思路：经典“移位相加” -> 量子版“受控写入部分积 + 加法器累加”
#
# |A>|B>|P=0>|T=0>|cin=0>  ->  |A>|B>|P=A*B>|T=0>|cin=0>
#
# 其中：
#  - A: n 位被乘数（保持不变）
#  - B: n 位乘数（保持不变）
#  - P: 2n 位积寄存器（输出）
#  - T: 2n 位临时寄存器（每轮写入一份“B_i 控制的 A<<i”部分积，随后加进 P，再反算清零）
#  - cin: 加法器进位输入位（始终保持 0，可复用）


from pyqpanda3.core import CPUQVM, QProg, QCircuit, X, CNOT, measure, TOFFOLI
from pyqpanda3.core.core import draw_qprog, PIC_TYPE


# =========================
# 1) 你的 CDKM/Cuccaro 加法器
#    语义：A 保持不变；B <- A + B (mod 2^n)
# =========================

def MAJ(c, b, a):
    """MAJ(c,b,a): 2*CNOT + 1*Toffoli"""
    cir = QCircuit()
    cir << CNOT(a, b)          # b ^= a
    cir << CNOT(a, c)          # c ^= a
    cir << TOFFOLI(b, c, a)    # a ^= b & c   -> a carries c_{i+1}
    return cir

def UMA(c, b, a):
    """UMA(c,b,a): 1*Toffoli + 2*CNOT"""
    cir = QCircuit()
    cir << TOFFOLI(b, c, a)    # undo
    cir << CNOT(a, c)          # restore c
    cir << CNOT(c, b)          # b ^= c   -> b becomes sum bit
    return cir

def cdkm_adder(A, B, cin, cout=None):
    """
    输入:
      A, B: list[int]，长度 n，小端序（A[0]最低位）
      cin:  int, 进位输入位（通常置 0）
      cout: int|None, 若给定则把最终进位 XOR 到 cout 上
    输出效果:
      A 还原不变；B <- A+B+cin (mod 2^n)；若 cout!=None 则 cout ^= carry_out
    """
    n = len(A)
    assert len(B) == n

    cir = QCircuit()

    # forward: ripple carry with MAJ
    cir << MAJ(cin, B[0], A[0])
    for i in range(1, n):
        cir << MAJ(A[i-1], B[i], A[i])

    # optional carry-out
    if cout is not None:
        cir << CNOT(A[n-1], cout)

    # backward: uncompute carries + write sums into B
    for i in range(n-1, 0, -1):
        cir << UMA(A[i-1], B[i], A[i])
    cir << UMA(cin, B[0], A[0])

    return cir


# =========================
# 2) 基础工具：装载整数（小端序）& 结果解析
# =========================

def load_int_into_reg(prog_or_cir, reg, value):
    """把非负整数 value 写进寄存器 reg（小端序），通过对为 1 的位加 X。"""
    for i, q in enumerate(reg):
        if (value >> i) & 1:
            prog_or_cir << X(q)

def bitstring_to_int_F(bitstr):
    """
    pyqpanda counts 的 key 通常按高位经典位 -> 低位经典位打印。
    如果我们 measure(q[i], i) 用的是小端序（i=0 是最低位），
    那把 bitstr 反转再按二进制解释，就得到对应整数。
    """
    return int(bitstr[::-1], 2)

def bitstring_to_int_R(bitstr):
    return int(bitstr, 2)
# =========================
# 3) 你要的：移位函数 lshift
# =========================

def lshift(reg, k, total_len=None, fill=None):
    """
    把“寄存器列表 reg（小端序）”逻辑左移 k 位，返回一个长度为 total_len 的列表：
      out[pos] = reg[pos-k]  (若 pos-k 在范围内)
      out[pos] = fill        (否则)

    例子：reg=[A0,A1,A2,A3], k=2, total_len=8
      out = [None,None,A0,A1,A2,A3,None,None]

    注意：这不是对量子态做门操作，只是“生成接线映射”。
    """
    if total_len is None:
        total_len = len(reg) + k
    out = [fill] * total_len
    for j, q in enumerate(reg):
        pos = j + k
        if 0 <= pos < total_len:
            out[pos] = q
    return out


# =========================
# 4) 量子乘法器：QMultiplier
# =========================

def QMultiplier(A, B, P, T, cin):
    """
    量子乘法器（移位相加法）：

    输入寄存器：
      A: n 位被乘数（保持不变）
      B: n 位乘数（保持不变）
      P: 2n 位积寄存器（初态应为 |0...0>，输出写到这里）
      T: 2n 位临时寄存器（初态应为 |0...0>，函数结束后会清回 0）
      cin: 1 位加法器进位输入（初态应为 |0>，函数结束后仍为 0）

    实现逻辑（第 i 轮）：
      1) 用 Toffoli 把 (B_i AND A) 写入 T 的移位位置：T ^= (A << i) * B_i
      2) 用加法器做：P <- P + T  (这里加法器的 A 输入是 T，B 输入是 P)
      3) 反算第 1) 步，把 T 清零，避免留下垃圾
    """
    n = len(A)
    assert len(B) == n
    assert len(P) == 2 * n
    assert len(T) == 2 * n

    cir = QCircuit()

    for i in range(n):
        # 生成 A 左移 i 位后的“接线视图”（长度 2n）
        A_shifted = lshift(A, i, total_len=2 * n, fill=None)

        # (1) T ^= (A<<i) * B[i] ：受控写入部分积
        # 对每一个有效位置 pos：若 A_shifted[pos] 存在，则 Toffoli(B[i], A_shifted[pos], T[pos])
        for pos, a_q in enumerate(A_shifted):
            if a_q is not None:
                cir << TOFFOLI(B[i], a_q, T[pos])

        # (2) 把 T 加到 P 上：P <- P + T  (mod 2^(2n))
        # 注意：cdkm_adder 会临时修改 T 但最终还原 T
        cir << cdkm_adder(T, P, cin, cout=None)

        # (3) 反算 (1)，清空 T
        for pos, a_q in enumerate(A_shifted):
            if a_q is not None:
                cir << TOFFOLI(B[i], a_q, T[pos])

    return cir


# =========================
# 5) 使用示例
# =========================

if __name__ == "__main__":
    n = 4

    # 按你原文件“直接用整数当 qubit index”的风格分配
    A = list(range(0, n))              # 0..3
    B = list(range(n, 2 * n))          # 4..7
    P = list(range(2 * n, 4 * n))      # 8..15  (2n 位积寄存器)
    T = list(range(4 * n, 6 * n))      # 16..23 (2n 位临时寄存器)
    cin = 6 * n                        # 24     (进位输入位，保持 0)

    a_val = 3
    b_val = 5
    expected = a_val * b_val

    prog = QProg()

    # 装载 A, B（P/T/cin 默认都是 0 不用动）
    load_int_into_reg(prog, A, a_val)
    load_int_into_reg(prog, B, b_val)

    # 乘法器电路：把 A*B 写入 P
    prog << QMultiplier(A, B, P, T, cin)

    # 测量 P：把 P[0..2n-1] 写到经典位 0..2n-1
    for i in range(2 * n):
        prog << measure(P[i], i)

    qvm = CPUQVM()
    qvm.run(prog, 1000)
    result = qvm.result().get_counts()

    # 可视化（生成 multiplier.tex）
    draw_qprog(prog, p=PIC_TYPE.LATEX, output_file="multiplier.tex")

    print(prog)
    print("Counts:", result)

    # 解析出现次数最多的结果，看看对应整数
    top_key = max(result, key=result.get)
    prod_val = bitstring_to_int_R(top_key)
    print(f"\nA={a_val}, B={b_val}, expected A*B={expected}")
    print(f"Most-likely measured P bits: {top_key}  ->  P={prod_val}")