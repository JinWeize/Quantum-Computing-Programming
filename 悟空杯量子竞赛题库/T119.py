from pyqpanda3.core.core import measure

import QFTFunction
from pyqpanda3.core import CPUQVM, QProg, QCircuit, H, CNOT, Encode
import math

prog = QProg()
qvm = CPUQVM()

qubits = [0, 1, 2]
x = [1 / math.sqrt(2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1 / math.sqrt(2)]

encode_circuit = Encode()
encode_circuit.amplitude_encode(qubits, x)

prog << encode_circuit.get_circuit()
prog << QFTFunction._qft_circuit(qubits)

qvm.run(prog, 1)
result1 = qvm.result().get_state_vector()
print(result1)

prog << measure(qubits[0], 0) << measure(qubits[1], 1) << measure(qubits[2], 2)
qvm.run(prog, 1000)
result2 = qvm.result().get_counts()
print(result2)