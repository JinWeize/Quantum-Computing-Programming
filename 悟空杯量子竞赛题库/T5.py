from pyqpanda3.core import QCircuit, CNOT, SWAP
from pyqpanda3.quantum_info import Unitary

cir0 = QCircuit()
cir0 << CNOT(0,1) << CNOT(0,1)
umatrix0 = Unitary(cir0)

cir1 = QCircuit()
cir1 << CNOT(0,1) << CNOT(1,0)
umatrix1 = Unitary(cir1)

cir2 = QCircuit()
cir2 << SWAP(0,1) << SWAP(0,1)
umatrix2 = Unitary(cir2)

cir3 = QCircuit()
cir3 << CNOT(1,0) << CNOT(1,0)
umatrix3 = Unitary(cir3)

print("A=B",umatrix0 == umatrix1)
print("B=C",umatrix1 == umatrix2)
print("B=D",umatrix1 == umatrix3)
print("C=D",umatrix2 == umatrix3)
print("A=D",umatrix0 == umatrix3)