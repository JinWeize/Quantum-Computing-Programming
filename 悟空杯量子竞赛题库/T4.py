from pyqpanda3.core import QCircuit, CNOT
from pyqpanda3.quantum_info import Unitary

cir0 = QCircuit()
cir0 << CNOT(0, 1) << CNOT(1, 2)
umatrix0 = Unitary(cir0)

cir1 = QCircuit()
cir1 << CNOT(1, 2) << CNOT(0, 1)
umatrix1 = Unitary(cir1)

cir2 = QCircuit()
cir2 << CNOT(0, 2) << CNOT(0, 1) << CNOT(1, 2)
umatrix2 = Unitary(cir2)

cir3 = QCircuit()
cir3 << CNOT(0, 1) << CNOT(1, 2) << CNOT(0, 2)
umatrix3 = Unitary(cir3)

print("A:", umatrix0 == umatrix1)
print("A:", umatrix2 == umatrix1)
print("A:", umatrix3 == umatrix1)
print("A:", umatrix2 == umatrix3)
