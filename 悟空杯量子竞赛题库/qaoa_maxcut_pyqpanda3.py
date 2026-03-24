"""
QAOA for MaxCut based on pyqpanda3.

This script follows the logic of the textbook example:
1. build a graph
2. prepare the uniform superposition state
3. apply p layers of phase-separator + mixer
4. sample the circuit
5. use a classical optimizer to maximize the expected cut value

Environment:
    pip install pyqpanda3 numpy scipy
Optional:
    pip install networkx matplotlib
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from pyqpanda3.core import QCircuit, QProg, H, RX, RZ, CNOT, measure, CPUQVM

try:
    import networkx as nx
except Exception:
    nx = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


Edge = Tuple[int, int]


# =========================
# 1. 问题定义与经典辅助函数
# =========================

def normalize_edges(edges: Iterable[Edge]) -> List[Edge]:
    """将边规范为 (min(u,v), max(u,v))，并去重。"""
    s = set()
    for u, v in edges:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        s.add((a, b))
    return sorted(s)


def maxcut_value_from_logical_bits(logical_bits: str, edges: Sequence[Edge]) -> int:
    """
    输入逻辑顺序 bit 串（第 i 位对应变量 x_i），返回 cut 的边数。
    比如 logical_bits='1010' 表示 x0=1,x1=0,x2=1,x3=0。
    """
    value = 0
    for u, v in edges:
        if logical_bits[u] != logical_bits[v]:
            value += 1
    return value


def measured_to_logical_bits(measured_bits: str) -> str:
    """
    pyqpanda/pyqpanda3 的测量结果字符串通常按高位经典位到低位经典位输出。
    为了让第 i 位对应 x_i，这里做一次翻转，和书中 state[::-1] 的处理一致。
    """
    return measured_bits[::-1]


def maxcut_value_from_measured_bits(measured_bits: str, edges: Sequence[Edge]) -> int:
    logical_bits = measured_to_logical_bits(measured_bits)
    return maxcut_value_from_logical_bits(logical_bits, edges)


def brute_force_maxcut(n: int, edges: Sequence[Edge]) -> Tuple[int, List[str]]:
    """经典暴力枚举，用于对照 QAOA 结果。"""
    best_value = -1
    best_solutions: List[str] = []
    for x in range(1 << n):
        bits = format(x, f"0{n}b")
        value = maxcut_value_from_logical_bits(bits, edges)
        if value > best_value:
            best_value = value
            best_solutions = [bits]
        elif value == best_value:
            best_solutions.append(bits)
    return best_value, best_solutions


# =========================
# 2. QAOA 量子线路
# =========================

def create_initial_state(n: int) -> QCircuit:
    """构造 |+>^n 初态。"""
    cir = QCircuit()
    for q in range(n):
        cir << H(q)
    return cir


def phase_separator_maxcut(n: int, edges: Sequence[Edge], gamma: float) -> QCircuit:
    r"""
    MaxCut 的相位分离层。

    MaxCut 的目标函数可写为
        C(z) = sum_{(u,v) in E} (1 - z_u z_v) / 2,   z_u in {+1, -1}
    对应的量子算符为
        H_C = sum_{(u,v) in E} (I - Z_u Z_v) / 2.

    QAOA 相位分离酉为
        U_C(gamma) = exp(-i * gamma * H_C).

    对于单条边 (u,v)，忽略全局相位后：
        exp[-i * gamma * (I - Z_u Z_v)/2] ~ exp(+i * gamma * Z_u Z_v / 2)
                                     = exp[-i * (-gamma) * Z_u Z_v / 2]

    而 CNOT(u,v) - RZ(v, theta) - CNOT(u,v)
    实现的是 exp[-i * theta * Z_u Z_v / 2]，因此这里取 theta = -gamma。
    """
    cir = QCircuit()
    for u, v in edges:
        cir << CNOT(u, v)
        cir << RZ(v, -gamma)
        cir << CNOT(u, v)
    return cir


def mixer_layer(n: int, beta: float) -> QCircuit:
    r"""
    混合层：U_B(beta) = exp(-i * beta * sum_i X_i)

    因为 RX(theta) = exp(-i * theta * X / 2)，
    所以这里每个量子比特上取 RX(2*beta)。
    """
    cir = QCircuit()
    for q in range(n):
        cir << RX(q, 2.0 * beta)
    return cir


def build_qaoa_circuit(n: int, edges: Sequence[Edge], gammas: Sequence[float], betas: Sequence[float]) -> QCircuit:
    if len(gammas) != len(betas):
        raise ValueError("gammas 和 betas 的长度必须相同。")

    cir = QCircuit()
    cir << create_initial_state(n)
    for gamma, beta in zip(gammas, betas):
        cir << phase_separator_maxcut(n, edges, gamma)
        cir << mixer_layer(n, beta)
    return cir


# =========================
# 3. 运行线路与计算目标函数
# =========================

def run_qaoa_counts(
    n: int,
    edges: Sequence[Edge],
    gammas: Sequence[float],
    betas: Sequence[float],
    shots: int = 2000,
    ) -> Dict[str, int]:
    """运行带测量的 QAOA 线路，返回测量频数。"""
    qvm = CPUQVM()
    prog = QProg()
    prog << build_qaoa_circuit(n, edges, gammas, betas)
    for q in range(n):
        prog << measure(q, q)

    qvm.run(prog, shots)
    counts = qvm.result().get_counts()
    return dict(counts)


def expected_cut_from_counts(counts: Dict[str, int], edges: Sequence[Edge]) -> float:
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0

    exp_val = 0.0
    for measured_bits, c in counts.items():
        exp_val += c * maxcut_value_from_measured_bits(measured_bits, edges)
    return exp_val / total_shots


class QAOAMaxCutSolver:
    def __init__(self, n: int, edges: Sequence[Edge], p: int = 1, shots: int = 2000, seed: int = 1):
        self.n = n
        self.edges = normalize_edges(edges)
        self.p = p
        self.shots = shots
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.eval_history: List[float] = []

    def objective(self, params: np.ndarray) -> float:
        """
        目标函数取“负的期望 cut 值”，
        因为 scipy.optimize.minimize 默认做最小化。
        """
        gammas = params[: self.p]
        betas = params[self.p :]
        counts = run_qaoa_counts(self.n, self.edges, gammas, betas, shots=self.shots)
        exp_cut = expected_cut_from_counts(counts, self.edges)
        self.eval_history.append(exp_cut)
        return -exp_cut

    def solve(self, method: str = "COBYLA", maxiter: int = 100):
        # 常见初始化：gamma 在 [0, pi]，beta 在 [0, pi/2]
        init_gammas = self.rng.uniform(0.0, math.pi, size=self.p)
        init_betas = self.rng.uniform(0.0, 0.5 * math.pi, size=self.p)
        x0 = np.concatenate([init_gammas, init_betas])

        result = minimize(
            self.objective,
            x0,
            method=method,
            options={"maxiter": maxiter},
        )

        best_params = result.x
        best_gammas = best_params[: self.p]
        best_betas = best_params[self.p :]
        final_counts = run_qaoa_counts(self.n, self.edges, best_gammas, best_betas, shots=self.shots)
        final_exp_cut = expected_cut_from_counts(final_counts, self.edges)

        ranked = sorted(final_counts.items(), key=lambda kv: kv[1], reverse=True)
        ranked_logical = [
            (measured_to_logical_bits(bits), count, maxcut_value_from_measured_bits(bits, self.edges))
            for bits, count in ranked
        ]

        return {
            "opt_result": result,
            "gammas": best_gammas,
            "betas": best_betas,
            "counts_measured": final_counts,
            "ranked_logical": ranked_logical,
            "expected_cut": final_exp_cut,
            "history": self.eval_history,
        }


# =========================
# 4. 可视化（可选）
# =========================

def plot_graph(n: int, edges: Sequence[Edge]) -> None:
    if nx is None or plt is None:
        print("未安装 networkx 或 matplotlib，跳过图可视化。")
        return

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)
    pos = nx.spring_layout(g, seed=7)
    nx.draw_networkx(g, pos=pos, with_labels=True)
    plt.title("MaxCut graph")
    plt.show()


def plot_counts(counts: Dict[str, int], title: str = "QAOA measurement counts") -> None:
    if plt is None:
        print("未安装 matplotlib，跳过频数图可视化。")
        return

    states = list(counts.keys())
    values = list(counts.values())
    plt.figure(figsize=(10, 4))
    plt.bar(states, values)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel("measured bitstrings")
    plt.ylabel("counts")
    plt.tight_layout()
    plt.show()


# =========================
# 5. 示例：书中 6 节点 MaxCut 图
# =========================

def main():
    # 与书中示例一致的一张 6 节点图
    n = 6
    edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5),
    ]

    # QAOA 层数
    p = 3
    shots = 2000

    print("=" * 60)
    print("QAOA for MaxCut (pyqpanda3)")
    print("n =", n)
    print("edges =", edges)
    print("p =", p, ", shots =", shots)
    print("=" * 60)

    plot_graph(n, edges)

    # 经典最优值，便于对照
    classical_best_value, classical_best_solutions = brute_force_maxcut(n, edges)
    print("经典暴力最优 cut 值:", classical_best_value)
    print("经典最优解（逻辑位串 x0x1...xn-1）:", classical_best_solutions)
    print()

    solver = QAOAMaxCutSolver(n=n, edges=edges, p=p, shots=shots, seed=1)
    result = solver.solve(method="COBYLA", maxiter=80)

    print("优化结束。")
    print("最优 gammas =", result["gammas"])
    print("最优 betas  =", result["betas"])
    print("最终采样估计的期望 cut 值 =", result["expected_cut"])
    print()

    print("按出现频数排序后的前 10 个候选解：")
    for logical_bits, count, cut_value in result["ranked_logical"][:10]:
        print(f"logical_bits = {logical_bits}, count = {count}, cut = {cut_value}")

    print()
    top_logical = result["ranked_logical"][0][0]
    print("QAOA 最常出现的逻辑位串 =", top_logical)
    print("其 cut 值 =", maxcut_value_from_logical_bits(top_logical, edges))

    plot_counts(result["counts_measured"], title="QAOA measurement counts (measured order)")


if __name__ == "__main__":
    main()
