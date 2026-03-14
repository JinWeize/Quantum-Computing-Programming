from pyqpanda3.core import CPUQVM, QProg, X, H, Z, QCircuit
from pyqpanda3.core.core import measure

qubits = [0, 1, 2]
query_qubits = qubits[:2]
ancilla = qubits[2]

qvm = CPUQVM()
prog = QProg()

def add_h(cir, qlist):
    for q in qlist:
        cir << H(q)
    return cir

def add_x(cir, qlist):
    for q in qlist:
        cir << X(q)
    return cir

oracle = QCircuit()
oracle << X(ancilla).control(query_qubits)
oracle << Z(ancilla)
oracle << X(ancilla).control(query_qubits)

diffuser = QCircuit()
add_h(diffuser, query_qubits)
add_x(diffuser, query_qubits)
diffuser << Z(query_qubits[-1]).control(query_qubits[:-1])
add_x(diffuser, query_qubits)
add_h(diffuser, query_qubits)

add_h(prog, query_qubits)

prog << oracle << diffuser

prog << measure(qubits[0], 0)
prog << measure(qubits[1], 1)

qvm.run(prog, 1000)
result = qvm.result().get_counts()

print(prog)
print(result)