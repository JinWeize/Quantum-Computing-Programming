from pyqpanda3.core import QCircuit, QProg, X, H, CNOT, measure, CPUQVM

cir0 = QCircuit()
cir1 = QCircuit()
cir2 = QCircuit()
cir3 = QCircuit()

cir0 << X(0) << X(2) << CNOT(1, 0) << CNOT(2, 1)
cir1 << X(0) << X(2) << CNOT(2, 1) << CNOT(1, 0)
cir2 << X(0) << X(2) << CNOT(2, 0) << CNOT(2, 1) << CNOT(1, 0)
cir3 << X(0) << X(2) << CNOT(2, 1) << CNOT(1, 0) << CNOT(2, 0)

prog0 = QProg()
prog1 = QProg()
prog2 = QProg()
prog3 = QProg()

prog0 << cir0
prog1 << cir1
prog2 << cir2
prog3 << cir3

prog0 << measure(0, 0) << measure(1, 1) << measure(2, 2)
prog1 << measure(0, 0) << measure(1, 1) << measure(2, 2)
prog2 << measure(0, 0) << measure(1, 1) << measure(2, 2)
prog3 << measure(0, 0) << measure(1, 1) << measure(2, 2)


qvm0 = CPUQVM()

qvm0.run(prog0, 1000)
result0 = qvm0.result().get_counts()
qvm0.run(prog1, 1000)
result1 = qvm0.result().get_counts()
qvm0.run(prog2, 1000)
result2 = qvm0.result().get_counts()
qvm0.run(prog3, 1000)
result3 = qvm0.result().get_counts()

print("A:", prog0, result0)
print("B:", prog1, result1)
print("C:", prog2, result2)
print("D:", prog3, result3)

