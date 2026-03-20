from pyqpanda3.core import CNOT, H, CPUQVM, QProg, QCircuit, CZ, measure, Encode
from pyqpanda3.quantum_info import Unitary
import numpy as np

from 悟空杯量子竞赛题库.T83 import cir_encode

print("=============第一问==============")
# 可以看出U_H就是H门
cir0 = QCircuit()
cir0 << H(0) << H(0)

print('U_H·U_H=:', Unitary(cir0))

print("=============第二问==============")
# U_{PS}(Pi) = CZ
x = '11'
qubts = [0, 1]
cir_encode = Encode()
cir_encode.basic_encode(qubts,x)

cir = cir_encode.get_circuit()
cir << H(1) << CZ(0, 1) << H(1)
prog = QProg(cir)
prog << measure(0, 0) << measure(1, 1)
qvm = CPUQVM()
qvm.run(prog, 1000)
result = qvm.result().get_counts()

print(result)

print("=============第三问==============")

x = '11'
qubts = [0, 1]
cir_encode = Encode()
cir_encode.basic_encode(qubts,x)

cir = cir_encode.get_circuit()
cir << H(1) << CNOT(0, 1) << H(1)
prog = QProg(cir)

qvm = CPUQVM()
qvm.run(prog, 1)
result = qvm.result().get_state_vector()
print(Unitary(cir))
print(result)
