import QFTFunction
from pyqpanda3.core import QProg, X, measure, CPUQVM
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

N = 11
x_value = 7
y_value = 5
z_value = 0
n = math.ceil(math.log(N, 2))

qvm = CPUQVM()
prog = QProg()

x_qubits = [0, 1, 2, 3]
y_qubits = [4, 5, 6, 7]
z_qubits = [8, 9, 10, 11]
aux = 12
work_msb = 13


prepare_basis_state_le(prog, x_qubits, x_value)
prepare_basis_state_le(prog, y_qubits, y_value)
prepare_basis_state_le(prog, z_qubits, z_value)

prog << QFTFunction.QFTVarModMul(x_qubits, y_qubits, z_qubits, work_msb, aux, N)
for i in range(n):
    prog << measure(x_qubits[i], i)
    prog << measure(y_qubits[i], i + n)
    prog << measure(z_qubits[i], i + 2 * n)
    prog << measure(work_msb, 1 + 3 * n)
    prog << measure(aux, 2 + 3 * n)

qvm.run(prog, 1000)
result = qvm.result().get_counts()

print(result)


