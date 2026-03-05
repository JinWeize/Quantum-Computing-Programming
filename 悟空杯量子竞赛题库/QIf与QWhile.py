# 参考：pyqpanda3 的 qif([cbit_idx]).then(...).qelse(...)
from pyqpanda3.core import CPUQVM, H, X, CNOT, QProg, qif, measure

qvm = CPUQVM()

prog = QProg()
true_branch = QProg()
false_branch = QProg()

# 先产生一个经典条件：把量子比特3测量到经典比特3
prog << X(3) << measure(3, 3)

# then 分支：H(0),H(1),H(2) 后测量 0,1,2
true_branch << H(0) << H(1) << H(2)
true_branch << measure([0, 1, 2], [0, 1, 2])

# else 分支：纠缠后测量 0,1,2
false_branch << H(0) << CNOT(0, 1) << CNOT(1, 2)
false_branch << measure([0, 1, 2], [0, 1, 2])

# 如果 c3 == 1 (True) 走 then，否则走 else
prog << qif([3]).then(true_branch).qelse(false_branch)

qvm.run(prog, 1000)
result = qvm.result().get_counts()

print(prog)
print(result)
