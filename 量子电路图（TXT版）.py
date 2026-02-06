import numpy as np

# 只保留本脚本实际需要的接口：measure / qif / 量子门 / QProg / QVM
from pyqpanda3.core import (
    QProg, CPUQVM,
    U3, RXX, RYY, RZZ, CNOT, BARRIER,
    measure, qif
)

# -----------------------------
# 论文代码.py 对应的结构实现
# -----------------------------

def first_layer(weights, wires):
    """对应论文代码 first_layer：每个qubit一个U3 + 邻接对上的 RXX/RYY/RZZ"""
    n = len(wires)

    # 每个qubit一组U3参数
    for i, q in enumerate(wires):
        U3_theta, U3_phi, U3_lam = map(float, weights[i*3:(i*3+3)])
        yield U3(q, U3_theta, U3_phi, U3_lam)

    # 后面是相邻耦合（每条边3个参数）
    nw = n * 3
    for p in [0, 1]:
        for indx, q in enumerate(wires):
            if indx % 2 == p and indx < n - 1:
                th_x, th_y, th_z = map(float, weights[nw:nw+3])
                q2 = wires[indx + 1]
                yield RXX(q, q2, th_x)
                yield RYY(q, q2, th_y)
                yield RZZ(q, q2, th_z)
                nw += 3


def convolutional_layer(weights, wires):
    """对应论文代码 convolutional_layer：两端U3 + RXX/RYY/RZZ"""
    n = len(wires)
    assert n >= 3, "this circuit is too small!"

    w = list(map(float, weights))
    for p in [0, 1]:
        for indx, q in enumerate(wires):
            if indx % 2 == p and indx < n - 1:
                q2 = wires[indx + 1]
                # 两个U3
                yield U3(q,  w[0], w[1], w[2])
                yield U3(q2, w[3], w[4], w[5])
                # 三个二体相互作用
                yield RXX(q, q2, w[6])
                yield RYY(q, q2, w[7])
                yield RZZ(q, q2, w[8])


def pooling_layer(weights, wires, cbits):
    """
    对应论文代码 pooling_layer：
    测量 odd qubit -> cbit，然后若测量为1，对前一个 even qubit 施加条件U3
    ⚠️ 这里包含动态控制流 qif，因此 draw_qprog 无法可视化，必须用 originir()
    """
    n = len(wires)
    assert n >= 2, "this circuit is too small!"

    w0, w1, w2 = map(float, weights)

    for indx, q in enumerate(wires):
        if indx % 2 == 1 and indx < n:
            c = cbits[q]  # 让 cbit 与 qubit 编号对应，方便管理
            yield measure(q, c)

            target = wires[indx - 1]  # 前一个 even qubit
            then_prog = QProg()
            then_prog << U3(target, w0, w1, w2)

            # ✅ pyqpanda3 动态电路：使用 qif 工厂函数（而不是 QIf 类）
            yield qif([c]).then(then_prog).qendif()


def conv_and_pooling(kernel_weights, wires, cbits, Cn=2):
    """
    对应论文代码 conv_and_pooling：
    kernel_weights 长度=3 + Cn*9
    前3个给 pooling，后面每9个给一个 convolutional_layer
    """
    kw = np.array(kernel_weights, dtype=float).ravel()
    assert len(kw) == 3 + Cn * 9

    # 先做 Cn 次卷积
    start = 3
    for _ in range(Cn):
        yield from convolutional_layer(kw[start:start+9], wires)
        start += 9

    # 再做池化（包含动态控制流）
    yield from pooling_layer(kw[:3], wires, cbits)


def dense_layer_2q(weights15, wires2):
    """
    论文里用的是 qml.ArbitraryUnitary(15参数)。
    这里用一个常见的“参数化两比特块”占位表示 dense 层结构：
    (U3,U3)-CNOT-(U3,U3)-CNOT-U3  共用15个参数。
    """
    assert len(wires2) == 2
    a, b = wires2
    w = list(map(float, np.array(weights15).ravel()))
    assert len(w) == 15

    yield U3(a, w[0], w[1], w[2])
    yield U3(b, w[3], w[4], w[5])
    yield CNOT(a, b)
    yield U3(a, w[6], w[7], w[8])
    yield U3(b, w[9], w[10], w[11])
    yield CNOT(a, b)
    yield U3(a, w[12], w[13], w[14])


# -----------------------------
# build_qcnn_prog（方案A：用 originir 导出）
# -----------------------------

def build_qcnn_prog(weights_21xL, last_layer_weights, num_wires=13, Cn=2, save_ir="qcnn_originir.txt"):
    """
    weights_21xL: shape=(21, layers) 其中 21=3+Cn*9 (Cn=2时=21)
    last_layer_weights: 长度应为 15 + (num_wires*3 + (num_wires-1)*3) = 15 + (6*num_wires-3)
                        num_wires=13 -> 90
    save_ir: 输出 originir 的文本文件路径（推荐 .txt）
    """
    qvm = CPUQVM()

    # 用整数编号表示 qubit/cbit
    wires = list(range(num_wires))
    cbits = list(range(num_wires))

    weights_21xL = np.array(weights_21xL, dtype=float)
    layers = weights_21xL.shape[1]

    last_layer_weights = np.array(last_layer_weights, dtype=float).ravel()
    expected = 15 + (6 * num_wires - 3)
    assert len(last_layer_weights) == expected, f"last_layer_weights长度应为{expected}，但你给的是{len(last_layer_weights)}"

    prog = QProg()

    # （可选）用一个Barrier占位表示“数据编码”
    prog << BARRIER(wires)

    # first_layer：用 last_layer_weights[15:]
    for op in first_layer(last_layer_weights[15:], wires):
        prog << op

    # conv + pooling 多层（含动态 qif）
    active_wires = wires[:]
    for j in range(layers):
        for op in conv_and_pooling(weights_21xL[:, j], active_wires, cbits, Cn=Cn):
            prog << op
        active_wires = active_wires[::2]
        prog << BARRIER(active_wires)

    # dense 层：最后应剩2个wire
    assert len(active_wires) == 2, f"池化后应剩2个qubit，但现在是{len(active_wires)}：{active_wires}"
    for op in dense_layer_2q(last_layer_weights[:15], active_wires):
        prog << op

    # 输出测量：测量 qubit0 -> cbit0
    prog << measure(0, 0)

    # ✅ 方案A：导出 IR（支持动态控制流）
    ir_text = prog.originir(precision=8)
    print("===== QProg ORIGINIR (preview) =====")
    print(ir_text)

    if save_ir:
        with open(save_ir, "w", encoding="utf-8") as f:
            f.write(ir_text)
        print(f"[OK] originir 已保存到：{save_ir}")

    return qvm, prog


if __name__ == "__main__":
    # 示例随机参数（你替换成自己的参数即可）
    weights_21x3 = np.random.rand(21, 3)
    last_layer_weights_90 = np.random.rand(90)

    qvm, prog = build_qcnn_prog(
        weights_21x3,
        last_layer_weights_90,
        num_wires=13,
        Cn=2,
        save_ir="qcnn_originir.txt"   # ✅ 输出 IR 到这个文件
    )

    # 如果你只是要“电路结构文本”，下面 run 不跑也可以
    qvm.run(prog, 1000)
    print(qvm.result().get_counts())
