from pyqpanda3.core import CZ, CNOT, H, QCircuit, QProg, CPUQVM
from pyqpanda3.quantum_info import Unitary

cir0 = QCircuit()
cir1 = QCircuit()
cir2 = QCircuit()
cir3 = QCircuit()
cir4 = QCircuit()

cir0 << CNOT(0, 1)
cir1 << H(1) << CZ(0, 1) << H(1)
cir2 << H(0) << CZ(0, 1) << H(1)
cir3 << H(1) << CZ(0, 1) << H(0)
cir4 << H(0) << CZ(0, 1) << H(1)

umatrix0 = Unitary(cir0)
umatrix1 = Unitary(cir1)
umatrix2 = Unitary(cir2)
umatrix3 = Unitary(cir3)
umatrix4 = Unitary(cir4)

print("A:", umatrix0 == umatrix1)
print("B:", umatrix0 == umatrix2)
print("C:", umatrix0 == umatrix3)
print("D:", umatrix0 == umatrix4)