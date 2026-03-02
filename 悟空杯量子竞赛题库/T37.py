from pyqpanda3.core import QCircuit, QProg, X, CNOT, measure, CPUQVM

cir = QCircuit()

cir << X(0) << X(2) << CNOT(1, 2) << CNOT(2, 1) << CNOT(1, 2) << CNOT(0, 1) << CNOT(1, 0) << CNOT(0, 1)

prog = QProg()
prog << cir
prog << measure(0,0) << measure(1,1)

qvm = CPUQVM()
qvm.run(prog, 1000)
result = qvm.result().get_counts()

print(result)
