import QFTFunction
from pyqpanda3.core import QProg, X, measure, CPUQVM

def prepare_basis_state_le(prog, qubits, value):
    """
    按小端序方式制备整数value。
    qubits[0] 为最低位。
    """
    for i, q in enumerate(qubits):
        if (value >> i) & 1:
            prog << X(q)
    return prog

x_value = 7
N_value = 11

n = 4

x_qubits = [0, 1, 2, 3]
aux = 4

qvm = CPUQVM()
prog = QProg()
prepare_basis_state_le(prog, x_qubits, x_value)

prog << QFTFunction.QFTVarModDou(x_qubits, aux, N_value)

prog << measure(x_qubits[0], 0)
prog << measure(x_qubits[1], 1)
prog << measure(x_qubits[2], 2)
prog << measure(x_qubits[3], 3)
prog << measure(aux, 4)

qvm.run(prog, 1000)
result = qvm.result().get_counts()

print(result)