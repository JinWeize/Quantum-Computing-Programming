from pyqpanda3.core import CPUQVM, QProg, QCircuit, X, CNOT, measure, TOFFOLI
from pyqpanda3.core.core import draw_qprog, PIC_TYPE


# =========================
#  1) 你原来已有的 CDKM 加法器
# =========================

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
      A, B: list[int]，长度 n，小端序（A[0] 最低位）
      cin:  int，进位输入位
      cout: int|None，若给定则把最终进位 XOR 到 cout 上

    输出效果:
      A 保持不变；B <- A + B (mod 2^n)
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


# =========================
#  2) 新增：受符号位控制的补码变换
# =========================

def ctrl_bitwise_not(ctrl, reg):
    """
    若 ctrl = 1，则对 reg 中每一位做按位取反；
    若 ctrl = 0，则 reg 不变。
    """
    cir = QCircuit()
    for q in reg:
        cir << CNOT(ctrl, q)
    return cir


def controlled_twos_complement_addsub(A, B, sign, cout=None):
    """
    受 sign 控制的补码加/减法器。

    约定：
      sign = 0  -> 计算 B <- A + B
      sign = 1  -> 计算 B <- A - B，对B取补码做减法，+sign=+cin即补码取反之后+1的操作
      用sign代替cin进行运算（二者等价）
    原理：
      sign = 0: B 不变，cin = 0，所以得到 A + B
      sign = 1: 先把 B 条件按位取反得到 ~B，再令 cin = 1，
                所以 adder 实际计算 A + (~B) + 1 = A - B

    输出：
      A 保持不变；B 被原位改写成结果（mod 2^n）
      若 cout != None：
         - 加法模式下，cout 是最高位进位
         - 减法模式下，cout = 1 表示“无借位”，cout = 0 表示“有借位”
      sign 在电路结束后恢复原值不变
    """
    n = len(A)
    assert len(B) == n

    cir = QCircuit()

    # sign=1 时，B -> ~B
    cir << ctrl_bitwise_not(sign, B)

    # 把 sign 本身当作 cin 使用：
    # sign=0 => +0
    # sign=1 => +1
    cir << cdkm_adder(A, B, sign, cout=cout)

    return cir


# =========================
#  3) 辅助函数
# =========================

def load_int_into_reg(prog_or_cir, reg, value):
    """把非负整数 value 写进寄存器 reg（小端序），通过对为 1 的位加 X。"""
    for i, q in enumerate(reg):
        if (value >> i) & 1:
            prog_or_cir << X(q)


def load_signed_magnitude(prog_or_cir, reg, sign, value):
    """
    把一个 Python 整数 value 写成“符号位 + 幅值位”的形式：
      value >= 0: sign = 0, reg = |value|
      value <  0: sign = 1, reg = |-value|

    注意：这里只是装载“符号-幅值表示”；
    真正的补码变换在 controlled_twos_complement_addsub() 内部完成。
    """
    mag = abs(value)
    if value < 0:
        prog_or_cir << X(sign)
    load_int_into_reg(prog_or_cir, reg, mag)


def bitstring_to_int_F(bitstr):
    """把测得的比特串（按 c[n-1]...c[0] 打印）转成整数，便于查看。"""
    return int(bitstr[::-1], 2)

def bitstring_to_int_R(bitstr):
    return int(bitstr, 2)
# =========================
#  4) 使用示例：A - B
# =========================
if __name__ == "__main__":
    n = 4

    # 量子位编号（延续你原文件的写法）
    A = list(range(0, n))          # 0,1,2,3
    B = list(range(n, 2 * n))      # 4,5,6,7
    sign = 2 * n                   # 8，控制“正/负”或“加/减”，此条只表示记录sign的量子比特序号
    cout = 2 * n + 1               # 9

    # 示例1：做减法 3 - 5 = -2 ≡ 14 (mod 16)
    a_val = 3
    b_val = 5
    sub_mode = 1                   # 1 表示减法；0 表示加法

    prog = QProg()

    # 初始化 A 和 B（都是幅值寄存器）
    load_int_into_reg(prog, A, a_val)
    load_int_into_reg(prog, B, b_val)

    # 设置sign位量子比特：1=减法，0=加法
    if sub_mode:
        prog << X(sign)

    # 受 sign 控制的补码加/减法器
    prog << controlled_twos_complement_addsub(A, B, sign, cout=cout)

    # 测量结果：B 为结果寄存器，cout 为最高位进/借位，sign 也可测出来确认未被破坏
    for i in range(n):
        prog << measure(B[i], i)
    prog << measure(cout, n)
    prog << measure(sign, n + 1)

    qvm = CPUQVM()
    qvm.run(prog, 1000)
    result = qvm.result().get_counts()

    # draw_qprog(prog, p=PIC_TYPE.LATEX, output_file="subtractor_twos_complement.tex")

    print(prog)
    print(result)

    # 解析若干测量结果
    print("\n解析（注意 counts 的键是高位经典位在左侧打印）：")
    for k, v in result.items():
        sign_bit = k[0]
        cout_bit = k[1]
        b_bits = k[2:]
        b_val_out = bitstring_to_int_R(b_bits)
        print(f"raw={k}, shots={v}, sign={sign_bit}, cout={cout_bit}, B_out={b_val_out}")
