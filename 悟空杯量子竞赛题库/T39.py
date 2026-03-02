from math import pi
from pyqpanda3.core import QCircuit, RY, CNOT, CPUQVM, QProg

circuit = QCircuit()

circuit << RY(0, pi / 2) << CNOT(0, 1) << CNOT(1, 2)

prog = QProg()
prog << circuit

qvm = CPUQVM()
qvm.run(prog, 1)
state = qvm.result().get_state_vector()

n = 3
eps = 1e-12
for i, amp in enumerate(state):
    if abs(amp) > eps:
        bitstr = format(i, f"0{n}b")
        print(bitstr, amp)