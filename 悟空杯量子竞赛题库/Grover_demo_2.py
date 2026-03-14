from pyqpanda3.core import CPUQVM, QProg, X, H, Z, QCircuit
from pyqpanda3.core.core import measure

qubits = [0, 1, 2, 3]
query_qubits = qubits[:3]   # x0, x1, x2
ancilla = qubits[3]         # 辅助比特

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

# target_bits 按 [x0, x1, x2] 给出，例如 [0,1,1] 表示 x0+x1+x2=2 中的 011
def mark_state(cir, target_bits):
    for i, bit in enumerate(target_bits):
        if bit == 0:
            cir << X(query_qubits[i])

    cir << X(ancilla).control(query_qubits)
    cir << Z(ancilla)
    cir << X(ancilla).control(query_qubits)

    for i, bit in enumerate(target_bits):
        if bit == 0:
            cir << X(query_qubits[i])

    return cir

# Oracle：查询条件 x0 + x1 + x2 = 2
# 对应目标态为 011, 101, 110
oracle = QCircuit()
mark_state(oracle, [0, 1, 1])
mark_state(oracle, [1, 0, 1])
mark_state(oracle, [1, 1, 0])

# 3 比特 diffuser
diffuser = QCircuit()
add_h(diffuser, query_qubits)
add_x(diffuser, query_qubits)
diffuser << Z(query_qubits[-1]).control(query_qubits[:-1])
add_x(diffuser, query_qubits)
add_h(diffuser, query_qubits)

# 初态制备
add_h(prog, query_qubits)

prog << oracle << diffuser

prog << measure(query_qubits[0], 0)
prog << measure(query_qubits[1], 1)
prog << measure(query_qubits[2], 2)

qvm.run(prog, 1000)
result = qvm.result().get_counts()

print(prog)
print(result)