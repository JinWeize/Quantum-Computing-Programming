#在PyQpanda3库中，已经有了CX门（CNOT）,以及CZ门（CZ），但是却没有CY门，在这里，我们利用QGate中的control方法，构造受控Y门（Contorled Y Gate, CY）
from pyqpanda3.core import QProg, QCircuit, Y, H, CPUQVM
from pyqpanda3.core.core import measure

cir = QCircuit()

cir << H(0) << Y(1).control(0)

prog = QProg()

# 统计概率
# prog << cir << measure(0, 0) << measure(1, 1)
#
# qvm = CPUQVM()
# qvm.run(prog, 1000)
# result = qvm.result().get_counts()
#
# print(prog)
# print(result)   #预测结果，应该为500 00，500 11

# 计算矢量

prog << cir

qvm = CPUQVM()
qvm.run(prog, 1)

state = qvm.result().get_state_vector()

n = 2
eps = 1e-12
for i, amp in enumerate(state):
    if abs(amp) > eps:
        bitstr = format(i, f"0{n}b")
        print(bitstr, amp)