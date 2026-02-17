from pyqpanda3.core import H, X, CNOT, QCircuit, QProg, CPUQVM
from pyqpanda3.core.core import measure

cir0 = QCircuit()
cir0 << X(1) << H(0) << CNOT(0, 1) << H(0) << H(1) << CNOT(0, 1)<< H(0) << H(1)

prog = QProg()
prog << cir0

prog << measure(0, 0) << measure(1,1)

qvm = CPUQVM()

qvm.run(prog, 2000)
result = qvm.result().get_counts()

print(prog)
print(result)