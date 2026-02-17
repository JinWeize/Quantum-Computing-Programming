from pyqpanda3.core import RX, RY, RZ, CNOT, QCircuit, QProg, CPUQVM
from pyqpanda3.quantum_info import Unitary
import math

cir0 = QCircuit()
cir1 = QCircuit()
cir2 = QCircuit()
cir3 = QCircuit()
cir4 = QCircuit()

cir0 << RY(0, math.pi / 4)
cir1 << RZ(0, math.pi) << RX(0, math.pi / 4) << RZ(0, math.pi / 2)
cir2 << RZ(0, math.pi / 2) << RX(0, math.pi / 4) << RZ(0, math.pi)
cir3 << RZ(0, math.pi / 4) << RX(0, math.pi / 2) << RZ(0, -math.pi / 2)
cir4 << RZ(0, math.pi / 2) << RX(0, math.pi / 4) << RZ(0, -math.pi / 2)

umatrix0 = Unitary(cir0)
umatrix1 = Unitary(cir1)
umatrix2 = Unitary(cir2)
umatrix3 = Unitary(cir3)
umatrix4 = Unitary(cir4)

print(umatrix0)
print(umatrix1)
print(umatrix2)
print(umatrix3)
print(umatrix4)