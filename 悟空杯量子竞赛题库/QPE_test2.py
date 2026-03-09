import QFTFunction
from pyqpanda3.core import QProg, X, measure, CPUQVM, QCircuit, U1
import math

def prepare_basis_state_le(prog, qubits, value):
    """
    按小端序方式制备整数value。
    qubits[0] 为最低位。
    """
    for i, q in enumerate(qubits):
        if (value >> i) & 1:
            prog << X(q)
    return prog

qvm = CPUQVM()
prog = QProg()
prog_estimate = QProg()
unitary_cir = QCircuit()
state_prep = QCircuit()

phase_qubits = [0, 1, 2, 3]
sys_qubits = [4]

unitary_cir << U1(4, math.pi / 4)
state_prep << X(sys_qubits[0])

cir = QFTFunction.QFTPhaseEstimation(phase_qubits, sys_qubits, unitary_cir, state_prep)

prog_estimate << cir
prog_estimate << measure(phase_qubits[0], 3)
prog_estimate << measure(phase_qubits[1], 2)
prog_estimate << measure(phase_qubits[2], 1)
prog_estimate << measure(phase_qubits[3], 0)

qvm.run(prog_estimate, 1000)
result = qvm.result().get_counts()

print(prog_estimate)
print(result)