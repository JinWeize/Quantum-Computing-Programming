import QFTFunction
from pyqpanda3.core import CPUQVM, QProg, QCircuit, X, H, Z, S, Encode, CR
import math
import numpy as np

prog = QProg()
qvm = CPUQVM()

qubits = [0, 1]
x = [1 / math.sqrt(14) + 0j, 3 / math.sqrt(14) + 0j, 2j / math.sqrt(14), 0 + 0j]

cir_encode = Encode()
# amplitude_encode编码顺序！！！太坑爹了，默认a0|00> + a1|10> + a2|01> + a3|11>
cir_encode.amplitude_encode(qubits, x)

prog << cir_encode.get_circuit()
prog << QFTFunction._qft_circuit(qubits)

qvm.run(prog, 1)
result = qvm.result().get_state_vector()

print(result)
