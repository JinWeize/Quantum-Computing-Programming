from pyqpanda3.core import CNOT, H, CPUQVM, QProg, QCircuit, CZ
from pyqpanda3.quantum_info import Unitary

print("=============第一问==============")
prog = QProg()
cir = QCircuit()

cir << CNOT(1, 0) << CNOT(1, 0)
prog << cir

print(prog)
print(Unitary(cir))
print("=============第二问==============")
cir0 = QCircuit()
cir1 = QCircuit()

cir0 << CNOT(1, 0)
cir1 << H(0) << H(1) << CNOT(0, 1) << H(0) << H(1)

print("两者相等",Unitary(cir0) == Unitary(cir1))

print("=============第三问==============")
cir2 = QCircuit()
cir3 = QCircuit()

cir2 << CNOT(0, 1)
cir3 << H(1) << CZ(0, 1) << H(1)
print(QProg(cir2))
print(QProg(cir3))
print("等价电路", Unitary(cir2) == Unitary(cir3))
