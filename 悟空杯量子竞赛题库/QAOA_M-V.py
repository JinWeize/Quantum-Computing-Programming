import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pyqpanda3.core import QCircuit, QProg, H, RX, RZ, CNOT, measure, CPUQVM
import math
seed = 1

def mv_to_ising(mu: np.ndarray, q: float, sigma: np.ndarray):
    """
    将M-V模型转化为Ising模型参数
    :param mu:随机n维向量
    :param q:风险系数
    :param sigma:某个随机n*n对称矩阵
    :return:const C常数项，h长度为n的列表或数组，J字典保存的值（i， j）
    """
    # 初始化变量
    n = len(mu)
    C = 0.0
    h = np.zeros((n, ), float)
    J = {}
    # 计算Ising参数
    for i in range(n):
        C = C + q * sigma[i, i] / 4 - mu[i]
        h[i] = mu[i] / 2
        for j in range(i, n):
            C = C + q * sigma[i, j] / 4
        for j in range(n):
            h[i] = h[i] - q / 4 * sigma[i, j]
            if i < j :
                J[(i, j)] = q * sigma[i, j] / 4

    return C, h, J


def build_cost_layer(qubits: list[int], h: list[float], J: dict, gamma: float) -> QCircuit:
    """
    建立问题哈密顿量，创建问题层量子电路。
    :param qubits: 量子比特序列
    :param h: Ising模型中一次项参数
    :param J: Ising模型中二次项参数
    :param gama: QAOA中cost layer参数
    :return: 量子电路cir
    """
    cir = QCircuit()
    for i in range(len(h)):
        cir << RZ(qubits[i], 2.0 * gamma * h[i])
    for (i, j), value in J.items():
        cir << CNOT(qubits[i], qubits[j]) << RZ(qubits[j], 2.0 * gamma * value) << CNOT(qubits[i], qubits[j])

    return cir


def build_mixer_layer(qubits: list[int], beta: float) -> QCircuit:
    """
    建立混合哈密顿量，创建混合层量子电路。
    :param qubits: 量子比特序列
    :param beta: mix layer中的参数
    :return: 量子电路cir
    """
    cir = QCircuit()
    for i in range(len(qubits)):
        cir << RX(qubits[i], 2.0 * beta)

    return cir


def bitstring_cost(bitstring: str, mu: np.ndarray, sigma: np.ndarray, q: float, n: int) -> float:
    """
    按照经典M-V公式计算某个bitstring的目标函数值。
    :param bitstring:测量得到的比特序列
    :param mu:参数
    :param sigma:参数
    :param q:参数
    :return:损失函数值
    """
    bitstring = bitstring[::-1]
    bit = np.zeros((n, ))
    for i in range(n):
        bit[i, ] = int(bitstring[i])
    U = 1 / 2 * q * bit.T @ sigma @ bit - mu.T @ bit

    return U


def compute_expectation(counts: dict, mu: np.ndarray, sigma: np.ndarray, q: float) -> float:
    """
    根据测量结果计算得到期望值，也就是损失函数。
    :param counts: 根据量子电路测量得到的结果
    :param mu: 参数，随机五维向量
    :param sigma: 随机5*5对称矩阵
    :param q: 风险系数
    :return: 期望值
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0

    exp_val = 0.0
    n_1 = len(mu)
    for measure_bits, c in counts.items():
        exp_val += c * bitstring_cost(measure_bits, mu, sigma, q, n_1)

    return exp_val / total_shots


def objective(params: np.ndarray, qubits: list[int], mu: np.ndarray, sigma: np.ndarray, q: float, p: int, shots: int) -> float:
    """
    根据当前参数求出期望值。
    :param params: 参数列表，前p个为gammas，后p个为betas
    :param mu: 五维随机参数向量
    :param sigma: 随机5*5对称矩阵
    :param q: 风险系数
    :param p: QAOA层数
    :param shots: 测量次数
    :return: 当前参数下的期望值
    """
    # 初始化参数
    gammas = params[: p: 1]
    betas = params[p: : 1]
    C, h, J = mv_to_ising(mu, q, sigma)
    # 构建量子虚拟机
    qvm = CPUQVM()
    cir = QCircuit()
    for i in range(len(mu)):
        cir << H(i)
    for gamma, beta in zip(gammas, betas):
        cir << build_cost_layer(qubits, h, J, gamma)
        cir << build_mixer_layer(qubits, beta)

    prog = QProg(cir)
    for i in range(len(qubits)):
        prog << measure(i, i)
    qvm.run(prog, shots)
    result = qvm.result().get_counts()

    exp = compute_expectation(result, mu, sigma, q)

    return exp


def evaluate_qaoa(params: np.ndarray, qubits: list[int], mu: np.ndarray, sigma: np.ndarray, q: float, p: int, shots: int) -> float:
    """
    根据最佳参数返回counts用来做分析。
    :param params: 参数列表，前p个为gammas，后p个为betas
    :param mu: 五维随机参数向量
    :param sigma: 随机5*5对称矩阵
    :param q: 风险系数
    :param p: QAOA层数
    :param shots: 测量次数
    :return: 当前参数下的期望值
    """
    # 初始化参数
    gammas = params[: p: 1]
    betas = params[p: : 1]
    C, h, J = mv_to_ising(mu, q, sigma)
    # 构建量子虚拟机
    qvm = CPUQVM()
    cir = QCircuit()
    for i in range(len(mu)):
        cir << H(i)
    for gamma, beta in zip(gammas, betas):
        cir << build_cost_layer(qubits, h, J, gamma)
        cir << build_mixer_layer(qubits, beta)

    prog = QProg(cir)
    for i in range(len(qubits)):
        prog << measure(i, i)
    qvm.run(prog, shots)
    result = qvm.result().get_counts()

    return result


def run_qaoa_mv(qubits: list[int], mu: np.ndarray, sigma: np.ndarray, q: float, p: int, shots: int, max_iter: int):
    """
    总控函数，调整参数，初始化参数，调用经典优化器minimize
    用最优参数运行线路，找到概率最大或者目标值最小的bitstring，返回结果。
    :param mu:
    :param sigma:
    :param q:
    :param p:
    :param shots:
    :param max_iter:
    :return:
    """
    n = len(mu)
    rng = np.random.default_rng(seed)
    init_gammas = rng.uniform(0.0, math.pi, size = p)
    init_betas = rng.uniform(0.0, 0.5 * math.pi, size = p)
    x0 = np.concatenate([init_gammas, init_betas])

    result = minimize(
        objective,
        x0,
        args = (qubits, mu, sigma, q, p, shots),
        method = "COBYLA",
        options={"maxiter":max_iter},
    )

    best_params = result.x
    best_gammas = best_params[: p]
    best_betas = best_params[p: ]
    final_counts = evaluate_qaoa(best_params, qubits, mu, sigma, q, p, shots)
    final_exp_cut = compute_expectation(final_counts, mu, sigma, q)

    ranked = sorted(final_counts.items(), key=lambda kv: kv[1], reverse = True)
    ranked_logical = [
        (bits[::-1], count, bitstring_cost(bits, mu, sigma, q, n))
        for bits, count in ranked
    ]

    return {
        "优化后结果": result,
        "gammas": best_gammas,
        "betas": best_betas,
        "counts_measured": final_counts,
        "ranked_logical": ranked_logical,
        "expected_value": final_exp_cut,
    }


def plot_counts(counts: dict, title: str = "M-V QAOA measurement counts") -> None:
    """
    根据最优参数测量得到的counts，绘制频率直方图。
    :param counts: 依据最优参数下量子电路测量运行得到的数据。
    :param title: 图表标题
    :return: 图表
    """
    states = list(counts.keys())
    values = list(counts.values())
    plt.figure(figsize = (10, 4))
    plt.bar(states, values)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel("measured bitstrings")
    plt.ylabel("counts")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 设定题目给定的参数
    n = 5
    mu = np.random.uniform(-0.02, 0.02, (n, ))
    sigma = np.random.uniform(-0.005, 0.005, (n, n))
    sigma = (sigma + sigma.T) / 2
    q = 5
    # QAOA层数
    p = 5
    shots = 20000
    max_iters = 100
    qubits = [0, 1, 2, 3, 4]
    # print(mv_to_ising(mu, q, sigma, n))
    # print(bitstring_cost('00001', mu, sigma, q, n))
    result = run_qaoa_mv(qubits, mu, sigma, q, p, shots,max_iters)
    # print(result)
    print("优化结束")
    print("\n最优 gammas = ", result["gammas"])
    print("\n最优 betas = ", result["betas"])
    print("\n最终采样估计的期望 损失值 =", result["expected_value"])

    print("\n按出现频数排序后的前 10 个候选解：")
    for logical_bits, count, value in result["ranked_logical"][:10]:
        print(f"logical_bits = {logical_bits}, count = {count}, value = {value}")

    # 可视化
    plot_counts(result['counts_measured'])

