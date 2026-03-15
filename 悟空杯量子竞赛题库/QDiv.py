# -*- coding: utf-8 -*-
"""
QDiv.py  ——  基于 CDKM/Cuccaro 加法器 + 补码减法思想 的“恢复除法( restoring division )”量子除法器

我们选择实现【最常见/用途最广】的“整数除法 + 余数”版本：
    给定 n 位无符号整数 dividend, divisor (divisor != 0)
    输出 quotient 与 remainder，使得：
        dividend = quotient * divisor + remainder, 且 0 <= remainder < divisor

为什么选“商 + 余数”：
- “除不尽”时余数是必须信息；并且 (dividend, divisor) -> (quotient, remainder, divisor) 这类映射更容易保持可逆/可逆电路思路；
- 在量子算法里，很多时候你其实只需要 remainder (模运算)；但做除法时同时得到 quotient/remainder 最通用。

实现特点：
- 不依赖 QWhile/QIf（避免不同版本 API 差异）；用固定 n 轮循环构造电路；
- dividend 寄存器会被【原位改写为 quotient】（节省寄存器、也避免“复制”导致纠缠）；
- remainder 单独用 (n+1) 位寄存器保存；
- divisor 保持不变；
- 使用“补码减法：B <- B - A = B + (~A) + 1”思想，并在减法后根据借位标志做“恢复”(add back)。

注意：
- 该实现面向“输入为计算基态(经典数装载)”的算术验证/练习场景最合适。
- 若输入处于叠加态，本电路依然是门级可逆电路，但会产生相应的量子相关性（这是量子算术的正常现象）。
"""

from pyqpanda3.core import CPUQVM, QProg, QCircuit, X, CNOT, measure, TOFFOLI


# ============================================================
# 1) 你的（上传减法器文件里）CDKM/Cuccaro ripple-carry 加法器
#    语义：A 不变；B <- A + B + cin  (mod 2^n)
# ============================================================

def MAJ(c, b, a):
    """MAJ(c,b,a): 2*CNOT + 1*Toffoli"""
    cir = QCircuit()
    cir << CNOT(a, b)          # b ^= a
    cir << CNOT(a, c)          # c ^= a
    cir << TOFFOLI(b, c, a)    # a ^= b & c
    return cir


def UMA(c, b, a):
    """UMA(c,b,a): 1*Toffoli + 2*CNOT"""
    cir = QCircuit()
    cir << TOFFOLI(b, c, a)    # undo
    cir << CNOT(a, c)          # restore c
    cir << CNOT(c, b)          # b ^= c -> b becomes sum bit
    return cir


def cdkm_adder(A, B, cin, cout=None):
    """
    输入:
      A, B: list[int]，长度 n，小端序（[0]最低位）
      cin:  int，进位输入位
      cout: int|None，若给定则把最终进位 XOR 到 cout 上

    输出效果:
      A 保持不变；B <- A + B + cin (mod 2^n)
      若 cout != None，则 cout ^= carry_out
    """
    n = len(A)
    assert len(B) == n

    cir = QCircuit()

    # forward: ripple carry with MAJ
    cir << MAJ(cin, B[0], A[0])
    for i in range(1, n):
        cir << MAJ(A[i - 1], B[i], A[i])

    # optional carry-out
    if cout is not None:
        cir << CNOT(A[n - 1], cout)

    # backward: uncompute carries + write sums into B
    for i in range(n - 1, 0, -1):
        cir << UMA(A[i - 1], B[i], A[i])
    cir << UMA(cin, B[0], A[0])

    return cir


# ============================================================
# 2) 基于补码思想实现 “B <- B - A” 的原位减法（A 还原不变）
#    B - A = B + (~A) + 1
#    我们只需：把 A 全位取反 -> 用加法器加到 B（cin=1）-> 再把 A 取反回来
# ============================================================

def sub_B_minus_A_inplace(A, B, one, cout=None):
    """
    计算：B <- B - A   (mod 2^n)
    其中 A, B 等长（小端序），A 最终还原不变。
    one: 一个常量 |1> 量子位，用作 cin=1
    cout: 输出“无借位标志”
        - cout=1 表示无借位（B >= A）
        - cout=0 表示有借位（B < A）
    """
    assert len(A) == len(B)

    cir = QCircuit()
    # A -> ~A
    for q in A:
        cir << X(q)
    # B <- (~A) + B + 1 = B - A
    cir << cdkm_adder(A, B, one, cout=cout)
    # 恢复 A
    for q in A:
        cir << X(q)
    return cir


# ============================================================
# 3) 受控版本加法器：用于“若借位则加回 divisor”(恢复步骤)
#    控制位 ctrl=1 时执行加法；ctrl=0 时什么都不做
#
#    关键：把加法器内部的
#       - CNOT 变为 Toffoli(ctrl, *, *)
#       - Toffoli 变为 3 控制门 CCCX(ctrl, *, *, *)
#    CCCX 用一个干净 anc（初态 |0>）分解实现
# ============================================================

def SWAP2(a, b):
    """用 3 个 CNOT 实现 SWAP(a,b)"""
    cir = QCircuit()
    cir << CNOT(a, b)
    cir << CNOT(b, a)
    cir << CNOT(a, b)
    return cir


def CCCX(ctrl, c1, c2, tgt, anc):
    """
    三控制X：当 ctrl=c1=c2=1 时翻转 tgt
    分解：anc 必须是 |0>
      anc ^= ctrl&c1
      tgt ^= anc&c2
      anc ^= ctrl&c1
    """
    cir = QCircuit()
    cir << TOFFOLI(ctrl, c1, anc)
    cir << TOFFOLI(c2, anc, tgt)
    cir << TOFFOLI(ctrl, c1, anc)
    return cir


def cMAJ(ctrl, c, b, a, anc):
    cir = QCircuit()
    # controlled CNOT(a,b)
    cir << TOFFOLI(ctrl, a, b)
    # controlled CNOT(a,c)
    cir << TOFFOLI(ctrl, a, c)
    # controlled TOFFOLI(b,c,a)
    cir << CCCX(ctrl, b, c, a, anc)
    return cir


def cUMA(ctrl, c, b, a, anc):
    cir = QCircuit()
    cir << CCCX(ctrl, b, c, a, anc)     # controlled TOFFOLI(b,c,a)
    cir << TOFFOLI(ctrl, a, c)          # controlled CNOT(a,c)
    cir << TOFFOLI(ctrl, c, b)          # controlled CNOT(c,b)
    return cir


def controlled_cdkm_adder(A, B, cin, ctrl, anc, cout=None):
    """
    受 ctrl 控制的加法器：
      ctrl=0：不改变任何寄存器
      ctrl=1：执行 B <- A + B + cin (mod 2^n)

    anc: |0> 的干净辅助位，用于分解 CCCX
    """
    n = len(A)
    assert len(B) == n

    cir = QCircuit()

    # forward
    cir << cMAJ(ctrl, cin, B[0], A[0], anc)
    for i in range(1, n):
        cir << cMAJ(ctrl, A[i - 1], B[i], A[i], anc)

    if cout is not None:
        # controlled CNOT(A[n-1], cout)
        cir << TOFFOLI(ctrl, A[n - 1], cout)

    # backward
    for i in range(n - 1, 0, -1):
        cir << cUMA(ctrl, A[i - 1], B[i], A[i], anc)
    cir << cUMA(ctrl, cin, B[0], A[0], anc)

    return cir


# ============================================================
# 4) 恢复除法（restoring division）核心：QDivider
#    dividend(Q) 原位改写为 quotient
#    remainder(R) 输出余数（(n+1) 位，通常最高位为 0）
#
#    每一轮 i = n-1 ... 0:
#      (a) R 左移 1，并把 Q[i] “下拉”到 R[0]（用 SWAP 实现，不复制不纠缠）
#      (b) 尝试：R <- R - D
#      (c) 若无借位：Q[i]=1（商位为1）
#          若有借位：恢复 R <- R + D，且 Q[i]=0
# ============================================================

def shift_left_1_into_R_from_Qbit(R, qbit, trash_bit):
    """
    将 R (m 位，小端序) 左移 1 位，并把 qbit 的值“下拉”到 R[0]。
    实现方式：
      1) SWAP(R[m-1],R[m-2]) ... SWAP(R[1],R[0])  (相当于把各位向高位移动一格)
      2) SWAP(R[0], trash_bit)  (trash_bit 初态为 |0>，把旧 R[0] 存入 trash_bit，同时令 R[0]=0)
      3) SWAP(R[0], qbit)       (把 dividend 的当前位移入 R[0]，同时 qbit 变为 0)
    """
    m = len(R)
    cir = QCircuit()

    for j in range(m - 1, 0, -1):
        cir << SWAP2(R[j], R[j - 1])

    cir << SWAP2(R[0], trash_bit)  # 令 R[0]=0，同时保留旧值到 trash
    cir << SWAP2(R[0], qbit)       # 把 qbit 移入 R[0]，qbit 变为 0
    return cir


def QDivider_restoring_inplace(Q, D_ext, R, one, cin0, anc, flags, trash):
    """
    恢复除法器（in-place）：

    输入：
      Q:     n 位 dividend（小端序），最终被改写为 quotient（小端序）
      D_ext: (n+1) 位 divisor（小端序，最高位通常为 0），保持不变
      R:     (n+1) 位 remainder（小端序），初态应为 |0...0>，输出余数
      one:   常量 |1> 位（用于减法时 cin=1）
      cin0:  常量 |0> 位（用于恢复加法时 cin=0）
      anc:   干净 |0> 辅助位（用于 CCCX 分解）
      flags: n 位辅助位，初态全 0。每轮存放“无借位标志”(cout)。
      trash: n 位辅助位，初态全 0。每轮存放 shift 时被丢弃的旧 R[0]，用于保证可逆。

    输出：
      Q 为 quotient
      R 为 remainder
      D_ext 不变
      flags/trash 会留下垃圾（若要彻底清理需进一步 uncompute；入门练习通常不必强求）
    """
    n = len(Q)
    assert len(flags) == n
    assert len(trash) == n
    assert len(D_ext) == n + 1
    assert len(R) == n + 1

    cir = QCircuit()

    # 从最高位到最低位做 n 轮
    for k in range(n):
        i = n - 1 - k  # i: 当前处理的 dividend 位（MSB -> LSB）

        # (a) R 左移 1，并把 Q[i] 下拉到 R[0]（Q[i] 被清为 0）
        cir << shift_left_1_into_R_from_Qbit(R, Q[i], trash[k])

        # (b) 尝试减法：R <- R - D_ext，flags[k] = no_borrow (1=成功, 0=借位)
        cir << sub_B_minus_A_inplace(D_ext, R, one, cout=flags[k])

        # (c) 若减法成功（no_borrow=1），则商位为 1：Q[i] ^= flags[k]
        cir << CNOT(flags[k], Q[i])

        # (d) 若借位（flags[k]=0），恢复：R <- R + D_ext
        #     用“借位标志 borrow = NOT(flags[k])”作为控制
        #     借位时：flags=0 -> X(flags)=1 -> 受控加法执行 -> 再 X 回来
        cir << X(flags[k])
        cir << controlled_cdkm_adder(D_ext, R, cin0, ctrl=flags[k], anc=anc, cout=None)
        cir << X(flags[k])

    return cir


# ============================================================
# 5) 辅助：装载整数到寄存器（计算基态），以及解析输出
# ============================================================

def load_int_into_reg(prog_or_cir, reg, value):
    """把非负整数 value 写进寄存器 reg（小端序），通过对为 1 的位加 X。"""
    for i, q in enumerate(reg):
        if (value >> i) & 1:
            prog_or_cir << X(q)


def parse_counts_key(key, n):
    """
    我们通常 measure(Q[i], i) / measure(R[j], n+j)。
    在很多 pyqpanda3 输出里，counts 的 key 是按 “高经典位 -> 低经典位” 打印。
    于是：
      key = c[2n] ... c[n]  c[n-1] ... c[0]
          = (R 的 (n+1) 位，从高到低) + (Q 的 n 位，从高到低)

    返回：(quotient, remainder)
    """
    # 右侧 n 位对应 Q 的 c[n-1]..c[0]
    q_bits = key[-n:]
    # 左侧 (n+1) 位对应 R 的 c[2n]..c[n]
    r_bits = key[:n+1]
    q_val = int(q_bits, 2)
    r_val = int(r_bits, 2)
    return q_val, r_val


# ============================================================
# 6) 使用示例
# ============================================================

if __name__ == "__main__":
    n = 4
    dividend_val = 13   # 1101
    divisor_val  = 3    # 0011

    # 量子位分配（延续你之前“用整数当 qubit index”的风格）
    Q = list(range(0, n))                 # dividend -> quotient
    D = list(range(n, 2 * n))             # divisor (n bits)
    d_msb0 = 2 * n                        # divisor 的扩展最高位（0）
    D_ext = D + [d_msb0]                  # (n+1) bits

    R = list(range(2 * n + 1, 2 * n + 1 + (n + 1)))  # remainder (n+1 bits)

    flags = list(range(3 * n + 2, 3 * n + 2 + n))    # n 个标志位
    trash = list(range(4 * n + 2, 4 * n + 2 + n))    # n 个 shift 垃圾位

    anc = 5 * n + 2     # CCCX 分解用的干净 anc（0）
    one = 5 * n + 3     # 常量 1（cin=1）
    cin0 = 5 * n + 4    # 常量 0（cin=0）

    # 构造程序
    prog = QProg()

    # 装载 dividend/divisor（计算基态）
    load_int_into_reg(prog, Q, dividend_val)
    load_int_into_reg(prog, D, divisor_val)
    prog << X(one)   # one 置为 |1>

    # 除法器
    prog << QDivider_restoring_inplace(Q, D_ext, R, one=one, cin0=cin0, anc=anc, flags=flags, trash=trash)

    # 测量：先测 Q 到 c0..c(n-1)，再测 R 到 c(n)..c(2n)
    for i in range(n):
        prog << measure(Q[i], i)
    for j in range(n + 1):
        prog << measure(R[j], n + j)

    qvm = CPUQVM()
    qvm.run(prog, 1000)
    counts = qvm.result().get_counts()

    print("Counts:", counts)

    top_key = max(counts, key=counts.get)
    q_out, r_out = parse_counts_key(top_key, n)
    print(f"dividend={dividend_val}, divisor={divisor_val}")
    print(f"measured key = {top_key}")
    print(f"quotient={q_out}, remainder={r_out}")
    print(f"check: quotient*divisor + remainder = {q_out*divisor_val + r_out}")
