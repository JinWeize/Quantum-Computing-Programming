from pyqpanda3.core import QProg, H, CNOT, measure, CPUQVM

# 1) 仅制备态：|psi> = (|00>+|11>)/sqrt(2) = (|++>+|-->)/sqrt(2)
prep = QProg()
prep << H(0) << CNOT(0, 1)

# 取状态矢量（不加测量更直观）
qvm = CPUQVM()
qvm.run(prep, 1)
sv = qvm.result().get_state_vector()
print("state_vector =", sv)   # 期望接近 [1/sqrt2, 0, 0, 1/sqrt2]
# QResult.get_state_vector() 接口见文档 :contentReference[oaicite:1]{index=1}

# 2) 在计算基测量验证：只会看到 00 和 11（各约 50%）
prog_z = QProg()
prog_z << H(0) << CNOT(0, 1) \
       << measure(0, 0) << measure(1, 1)

qvm.run(prog_z, 2000)
counts_z = qvm.result().get_counts()
print("Z-basis counts =", counts_z)
# CPUQVM.run / result / get_counts 见文档 :contentReference[oaicite:2]{index=2}

# 3) （可选）在 X 基测量验证：测 X 等价于先 H 再按 Z 测
# 结果 00 对应 ++，11 对应 --
prog_x = QProg()
prog_x << H(0) << CNOT(0, 1) \
       << H(0) << H(1) \
       << measure(0, 0) << measure(1, 1)

qvm.run(prog_x, 2000)
counts_x = qvm.result().get_counts()
print("X-basis counts (after H) =", counts_x)
