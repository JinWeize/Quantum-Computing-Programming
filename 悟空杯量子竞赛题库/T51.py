from pyqpanda3.core import QProg, H, CNOT, X, measure, CPUQVM, QCircuit

cir = QCircuit()
cir << H(0) << X(1) << CNOT(0, 1) << H(0) << H(1) << CNOT(0, 1) << H(0) << H(1)

prog = QProg()
prog << cir

qvm = CPUQVM()
qvm.run(prog, 1)

state = qvm.result().get_state_vector()

n = 2
eps = 1e-12
for i, amp in enumerate(state):
    if abs(amp) > eps:
        bitstr = format(i, f"0{n}b")
        print(bitstr, amp)

