from pyqpanda3.core import QCircuit, QProg, H, CNOT, measure, CPUQVM

# 创建一个量子线路
circuit = QCircuit()

# 构造 GHZ 态
circuit << H(0)  # 在比特 0 上应用 Hadamard 门
circuit << CNOT(0, 1)  # 以 0 号比特为控制，1 号比特为目标，应用 CNOT 门
circuit << CNOT(1, 2)  # 以 1 号比特为控制，2 号比特为目标，应用 CNOT 门

# 创建量子程序并组合线路
prog = QProg()
prog << circuit

# 添加测量操作
prog << measure(0, 0) << measure(1, 1) << measure(2, 2)

# 创建 QVM
qvm = CPUQVM()

# 运行量子程序并获取结果
qvm.run(prog, 1000)
result = qvm.result().get_counts()

# 打印量子程序和结果
print(prog)