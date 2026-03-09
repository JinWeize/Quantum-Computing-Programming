# 利用ChatGPT辅助编写
# cdkm_adder为基于非多数加法（Unmajority and Add, UMA）与就地多数(in-place Majority，MAJ)的Cuccaro等（2024arXiv）论文的量子加法器

from pyqpanda3.core import CPUQVM, QProg, QCircuit, X, CNOT, measure, TOFFOLI
from pyqpanda3.core.core import draw_qprog, PIC_TYPE


# Toffoli 在多数版本里叫 Toffoli；如果你环境里名字不同，可把下面 try/except 换成你实际的门名
# try:
#     from pyqpanda3.core import Toffoli
# except Exception:
#     # 兜底：用“给 X 加两个控制”构造 Toffoli（若你的版本支持 .control）
#     from pyqpanda3.core import X as _X
#     def Toffoli(c1, c2, t):
#         return _X(t).control([c1, c2])

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
      A 还原不变；B <- A+B (mod 2^n)；若 cout!=None 则 cout ^= carry_out
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

def load_int_into_reg(prog_or_cir, reg, value):
    """把整数 value 写进寄存器 reg（小端序），通过对为 1 的位加 X。"""
    for i, q in enumerate(reg):
        if (value >> i) & 1:
            prog_or_cir << X(q)

# 使用案例：
n = 4
A = list(range(0, n))        # 0,1,2,3
B = list(range(n, 2*n))      # 4,5,6,7
cin = 2*n                    # 8
cout = 2*n + 1               # 9

a_val = 3
b_val = 5

prog = QProg()

# 初始化 A,B
load_int_into_reg(prog, A, a_val)
load_int_into_reg(prog, B, b_val)

# 加法器电路
prog << cdkm_adder(A, B, cin, cout=cout)

# 测量：把 B 写到 c0..c3，把 cout 写到 c4
for i in range(n):
    prog << measure(B[i], i)
prog << measure(cout, n)

qvm = CPUQVM()
qvm.run(prog, 1000)
result = qvm.result().get_counts()

# print(PIC_TYPE.__members__)
draw_qprog(prog, p = PIC_TYPE.LATEX, output_file="adder.tex")
print(prog)
print(result)