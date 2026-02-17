from pyqpanda3.core import QCircuit, CNOT, SWAP, X, Y, Z, S, QProg
from pyqpanda3.quantum_info import Unitary

cir0 = QCircuit()
cir0 << X(1) << CNOT(0, 1) << Y(1) << CNOT(0,1) << S(0) << Z(1)

prog = QProg()
prog << cir0


qmatrix = Unitary(cir0)
print(qmatrix)
print(prog)