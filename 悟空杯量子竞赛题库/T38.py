from pyqpanda3.core import QCircuit, QProg, X, H, CNOT, measure, CPUQVM

cir = QCircuit()
cir << X(0) << X(1) << X(2) << H(0) << H(1) << H(2) << H(3) << CNOT(3, 2) << H(2) << CNOT(2, 1) << H(1) << CNOT(1, 0) << H(0)

prog = QProg()
prog << cir
prog << measure(0, 0) << measure(1, 1) << measure(2, 2) << measure(3, 3)

qvm = CPUQVM()
qvm.run(prog, 1000)
result = qvm.result().get_counts()

print(prog)
print(result)


cir1 = QCircuit(4)
cir1 << X(0) << X(1) << X(2) \
    << H(0) << H(1) << H(2) << H(3) \
    << CNOT(3, 2) << H(2) \
    << CNOT(2, 1) << H(1) \
    << CNOT(1, 0) << H(0)

prog_sv = QProg()
prog_sv << cir1

qvm1 = CPUQVM()
qvm1.run(prog_sv, 1)

state = qvm1.result().get_state_vector()  # list[complex]  :contentReference[oaicite:1]{index=1}

n = 4
eps = 1e-12
for i, amp in enumerate(state):
    if abs(amp) > eps:
        bitstr = format(i, f"0{n}b")
        print(bitstr, amp)