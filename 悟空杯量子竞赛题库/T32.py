from pyqpanda3.core import CNOT, H, Z, QCircuit, QProg, measure,CPUQVM

circuit = QCircuit()

circuit << H(0) << CNOT(0, 1) << Z(0)

prog = QProg()
prog << circuit
prog << H(0) << H(1)
prog << measure(0, 0) <<measure(1, 1)

qvm = CPUQVM()

qvm.run(prog, 1000)
result = qvm.result().get_counts()

print(prog)
print(result)