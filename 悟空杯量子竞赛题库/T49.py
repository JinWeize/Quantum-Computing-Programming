from pyqpanda3.core import QCircuit, CNOT, X, Y
from pyqpanda3.quantum_info import Unitary

cir0 = QCircuit()
cir0 << CNOT(0, 1) << X(0) << CNOT(0, 1)
umatrix0 = Unitary(cir0)

cir1 = QCircuit()
cir1 << CNOT(0, 1) << Y(0) << CNOT(0, 1)
umatrix1 = Unitary(cir1)

cir2 = QCircuit()
cir2 << X(0) << X(1)
umatrix2 = Unitary(cir2)

cir3 = QCircuit()
cir3 << Y(0) << X(1)
umatrix3 = Unitary(cir3)

cir4 = QCircuit()
cir4 << X(1)
umatrix4 = Unitary(cir4)

cir5 = QCircuit()
cir5 << Y(0)
umatrix5 = Unitary(cir5)

print("A:", umatrix0 == umatrix2)
print("B:", umatrix1 == umatrix3)
print("C:", umatrix0 == umatrix4)
print("D:", umatrix1 == umatrix5)