from pyqpanda3.core import CNOT, QProg, QCircuit, TOFFOLI
from pyqpanda3.quantum_info import Unitary

cir0 = QCircuit()
cir1 = QCircuit()
cir2 = QCircuit()
cir3 = QCircuit()

#小端比特序，第一位量子比特与第三位量子比特互换存储位置
cir0 << TOFFOLI(1, 0, 2) << CNOT(0, 1) << CNOT(1, 0) << CNOT(2, 1) << TOFFOLI(2, 1, 0) << CNOT(1, 2)
cir1 << TOFFOLI(1, 0, 2) << CNOT(0, 1) << CNOT(1, 0) << CNOT(2, 1) << TOFFOLI(2, 1, 0) << CNOT(2, 1)
cir2 << TOFFOLI(1, 0, 2) << CNOT(0, 1) << CNOT(1, 0) << CNOT(1, 2) << TOFFOLI(2, 1, 0) << CNOT(2, 1)
cir3 << TOFFOLI(2, 1, 0) << CNOT(0, 1) << CNOT(1, 0) << CNOT(2, 1) << TOFFOLI(2, 1, 0) << CNOT(2, 1)

qmatrix0 = Unitary(cir0)
qmatrix1 = Unitary(cir1)
qmatrix2 = Unitary(cir2)
qmatrix3 = Unitary(cir3)

# qprog = QProg(cir1)
# print(qprog)

print("A:", qmatrix0)
print("B:", qmatrix1)
print("C:", qmatrix2)
print("D:", qmatrix3)