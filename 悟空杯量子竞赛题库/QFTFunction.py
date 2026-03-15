import math
from pyqpanda3.core import QCircuit, H, X, CR, CNOT, SWAP, U1, TOFFOLI

__all__ = [
    "AMULN",
    "ADDN",
    "QFTConAdd",
    "ConMULTN",
    "QFTConModExp",
    "QFTAdder",
    "QFTVarModAdd",
    "QFTVarModDou",
    "QFTVarModMul",
    "QFTVarModSqr",
    "QFTPhaseEstimation",
    "_qft_circuit"
]


# 内部基础模块：QFT / IQFT / 角度计算

"""
[MSB……LSB]
[q_0, ……， q_1]
"""
def _qft_circuit(qubits, with_swap=False):
    """
    构造 QFT 电路。

    参数中的 qubits 采用算术顺序 [MSB, ..., LSB]。
    这里默认 with_swap=False，便于与后续常数加法器的位序保持一致。
    """
    cir = QCircuit()
    n = len(qubits)

    for i in range(n):
        target = qubits[i]
        cir << H(target)
        for j in range(i + 1, n):
            control = qubits[j]
            angle = math.pi / (2 ** (j - i))
            cir << CR(control, target, angle)

    if with_swap:
        for i in range(n // 2):
            cir << SWAP(qubits[i], qubits[n - 1 - i])

    return cir



def _iqft_circuit(qubits, with_swap=False):
    """
    构造逆 QFT 电路。

    参数中的 qubits 采用算术顺序 [MSB, ..., LSB]。
    """
    cir = QCircuit()
    n = len(qubits)

    if with_swap:
        for i in range(n // 2):
            cir << SWAP(qubits[i], qubits[n - 1 - i])

    for i in range(n - 1, -1, -1):
        target = qubits[i]
        for j in range(n - 1, i, -1):
            control = qubits[j]
            angle = -math.pi / (2 ** (j - i))
            cir << CR(control, target, angle)
        cir << H(target)

    return cir



def _get_angles(a, n):
    """
    计算基于 QFT 的常数加法器所需相位数组。

    这里适配 _qft_circuit(..., with_swap=False)，并假定 qubits 采用
    [MSB, ..., LSB] 的顺序。返回的 angles 与传入寄存器顺序一一对应。
    """
    a = a % (1 << n)
    bits = [(a >> k) & 1 for k in range(n)]  # bits[0] 为最低位

    angles = []
    for i in range(n):
        le_idx = n - 1 - i
        theta = 0.0
        for k in range(le_idx + 1):
            if bits[k]:
                theta += 2 * math.pi / (2 ** (le_idx - k + 1))
        angles.append(theta)

    return angles


# 内部辅助模块：受控加法 / 受控模加 / 模逆


def _ctrl_qft_con_add(qubits, ctrl, a):
    """
    基于 QFT 的受控常数加法器。

    当 ctrl=1 时：|x> -> |x + a (mod 2^n)
    当 ctrl=0 时：不变

    参数:
        qubits: 算术顺序 [MSB, ..., LSB]
    """
    cir = QCircuit()
    angles = _get_angles(a, len(qubits))

    cir << _qft_circuit(qubits, with_swap=False)
    for q, theta in zip(qubits, angles):
        cir << CR(ctrl, q, theta)
    cir << _iqft_circuit(qubits, with_swap=False)

    return cir



def _ctrl_qft_con_mod_add(x_qubits, work_msb, aux, ctrl, a, N):
    """
    受控常数模加法器。

    当 ctrl=1 时，|x> -> |(x + a) mod N>
    当 ctrl=0 时，|x> 不变
    """
    cir = QCircuit()
    n = len(x_qubits)

    if not (0 <= a < N):
        raise ValueError("要求满足 0 <= a < N")
    if N >= (1 << n):
        raise ValueError("要求模数 N 小于 2^n，x_qubits 位数不足")

    full_reg = [work_msb] + list(reversed(x_qubits))

    # 1) 当 ctrl=1 时计算 x + a - N
    cir << _ctrl_qft_con_add(full_reg, ctrl, a - N)
    # 2) 用 aux 记录 ctrl AND work_msb
    cir << TOFFOLI(ctrl, work_msb, aux)
    # 3) 若 aux=1，则加回 N
    cir << _ctrl_qft_con_add(full_reg, aux, N)
    # 4) 反计算 aux
    cir << _ctrl_qft_con_add(full_reg, ctrl, -a)
    cir << CNOT(ctrl, aux)
    cir << TOFFOLI(ctrl, work_msb, aux)
    cir << _ctrl_qft_con_add(full_reg, ctrl, a)

    return cir



def _egcd(a, b):
    """扩展欧几里得算法。"""
    if b == 0:
        return a, 1, 0
    g, x1, y1 = _egcd(b, a % b)
    return g, y1, x1 - (a // b) * y1



def _modinv(a, N):
    """求 a 在模 N 下的逆元，要求 gcd(a, N) = 1。"""
    g, x, _ = _egcd(a, N)
    if g != 1:
        raise ValueError("a 与 N 不互素，模逆元不存在，无法构造可逆的常数模乘法器")
    return x % N


# 对外核心函数

def QFTConAdd(qubits, a):
    """
    基于 QFT 的常数加法器。

    实现:
        |x> -> |x + a (mod 2^n)>

    参数:
        qubits: 算术顺序 [MSB, ..., LSB]
        a:      要加的常数（允许传入负数，内部按模 2^n 处理）

    返回:
        QCircuit
    """
    cir = QCircuit()
    angles = _get_angles(a, len(qubits))

    cir << _qft_circuit(qubits, with_swap=False)
    for q, theta in zip(qubits, angles):
        cir << U1(q, theta)
    cir << _iqft_circuit(qubits, with_swap=False)

    return cir



def ADDN(x_qubits, work_msb, aux, a, N):
    """
    基于 QFT 的常数模加法器。

    实现:
        |x> -> |(x + a) mod N>

    参数:
        x_qubits : 数据寄存器，little-endian（x_qubits[0] 为最低位）
        work_msb : 工作寄存器最高位 / 借位位，初态需为 |0>
        aux      : 辅助位，初态需为 |0>
        a        : 常数加数，要求 0 <= a < N
        N        : 模数，要求 N < 2^n

    返回:
        QCircuit
    """
    cir = QCircuit()
    n = len(x_qubits)

    if not (0 <= a < N):
        raise ValueError("要求满足 0 <= a < N")
    if N >= (1 << n):
        raise ValueError("要求模数 N 小于 2^n，x_qubits 位数不足")

    # QFT 模块内部统一采用 [MSB, ..., LSB]
    full_reg = [work_msb] + list(reversed(x_qubits))

    # 1) x + a - N
    cir << QFTConAdd(full_reg, a - N)
    # 2) 取最高位到 aux
    cir << CNOT(work_msb, aux)
    # 3) 若 aux=1，则加回 N
    cir << _ctrl_qft_con_add(full_reg, aux, N)
    # 4) 反计算 aux
    cir << QFTConAdd(full_reg, -a)
    cir << X(aux)
    cir << CNOT(work_msb, aux)
    cir << QFTConAdd(full_reg, a)

    return cir



def AMULN(x_qubits, y_qubits, work_msb, aux, a, N):
    """
    基于 QFT 的常数模加乘运算器。

    实现:
        |x>|y> -> |x>|(y + a*x) mod N>

    参数:
        x_qubits : 乘数寄存器，little-endian（x_qubits[0] 为最低位）
        y_qubits : 被累加寄存器，little-endian（y_qubits[0] 为最低位）
        work_msb : 作用在 y 寄存器上的工作最高位 / 借位位，初态需为 |0>
        aux      : 辅助位，初态需为 |0>
        a        : 常数，要求 0 <= a < N
        N        : 模数，要求 N < 2^n

    返回:
        QCircuit
    """
    cir = QCircuit()
    n = len(x_qubits)

    if len(y_qubits) != n:
        raise ValueError("这里要求 x_qubits 与 y_qubits 位数一致")
    if not (0 <= a < N):
        raise ValueError("要求满足 0 <= a < N")
    if N >= (1 << n):
        raise ValueError("要求模数 N 小于 2^n，寄存器位数不足")

    for j, ctrl in enumerate(x_qubits):
        add_const = (a * (1 << j)) % N
        cir << _ctrl_qft_con_mod_add(y_qubits, work_msb, aux, ctrl, add_const, N)

    return cir



def ConMULTN(x_qubits, y_qubits, work_msb, aux, a, N):
    """
    基于 QFT 的常数模乘法器。

    实现:
        在 y 寄存器初态为 |0> 时，完成
            |x>|0> -> |a*x mod N>|0>

    参数:
        x_qubits : 输入 / 输出寄存器，little-endian
        y_qubits : 辅助结果寄存器，little-endian，要求初态为 |0>
        work_msb : 作用在加法目标寄存器上的工作最高位 / 借位位，初态需为 |0>
        aux      : 辅助位，初态需为 |0>
        a        : 常数，要求 0 <= a < N 且 gcd(a, N)=1
        N        : 模数，要求 N < 2^n

    返回:
        QCircuit
    """
    cir = QCircuit()
    n = len(x_qubits)

    if len(y_qubits) != n:
        raise ValueError("这里要求 x_qubits 与 y_qubits 位数一致")
    if not (0 <= a < N):
        raise ValueError("要求满足 0 <= a < N")
    if N >= (1 << n):
        raise ValueError("要求模数 N 小于 2^n，寄存器位数不足")

    a_inv = _modinv(a, N)

    # 1) |x>|0> -> |x>|a*x mod N>
    cir << AMULN(x_qubits, y_qubits, work_msb, aux, a, N)

    # 2) 交换两个寄存器
    for qx, qy in zip(x_qubits, y_qubits):
        cir << SWAP(qx, qy)

    # 3) 清空第二寄存器：y <- y - a_inv * x (mod N)
    cir << AMULN(x_qubits, y_qubits, work_msb, aux, (-a_inv) % N, N)

    return cir



def QFTConModExp(exp_qubits, acc_qubits, tmp_qubits, work_msb, aux, a, N):
    """
    基于 QFT 的常数模幂运算器。

    实现:
        在 acc_qubits 初态为 |1>、tmp_qubits 初态为 |0> 时，完成
            |x>|1>|0> -> |x>|a^x mod N>|0>

    其中:
        x = sum_j exp_qubits[j] * 2^j

    参数:
        exp_qubits : 指数寄存器，little-endian（exp_qubits[0] 为最低位）
        acc_qubits : 累乘寄存器 / 结果寄存器，little-endian，初态需为 |1>
        tmp_qubits : 模乘辅助寄存器，little-endian，位数需与 acc_qubits 一致，初态需为 |0>
        work_msb   : 供模加/模乘内部使用的工作最高位 / 借位位，初态需为 |0>
        aux        : 辅助位，初态需为 |0>
        a          : 常数底数，要求 gcd(a, N) = 1
        N          : 模数，要求 N < 2^n，其中 n = len(acc_qubits)

    返回:
        QCircuit
    """
    cir = QCircuit()

    n = len(acc_qubits)

    if len(tmp_qubits) != n:
        raise ValueError("要求 acc_qubits 与 tmp_qubits 位数一致")
    if N <= 0:
        raise ValueError("要求模数 N 为正整数")
    if N >= (1 << n):
        raise ValueError("要求模数 N 小于 2^n，寄存器位数不足")

    a = a % N

    if math.gcd(a, N) != 1:
        raise ValueError("要求 gcd(a, N) = 1，否则常数模乘不可逆，无法构造常数模幂运算器")

    # 逐位处理指数寄存器：
    # 若 exp_qubits[j] = 1，则对 acc_qubits 施加一次
    # 乘以 a^(2^j) mod N 的受控常数模乘。
    for j, ctrl in enumerate(exp_qubits):
        mul_const = pow(a, 1 << j, N)

        # 乘以 1 等于恒等变换，可直接跳过
        if mul_const == 1:
            continue

        mul_cir = ConMULTN(acc_qubits, tmp_qubits, work_msb, aux, mul_const, N)

        # 将整段常数模乘线路提升为受控线路
        cir << mul_cir.control([ctrl])

    return cir



def QFTAdder(x_qubits, y_qubits, aux1, aux0):
    """
    基于 QFT 的量子变量加法器（out-of-place）。

    实现:
        |x>|y>|0...0>|0>  ->  |x>|y>|x+y (low n bits)>|carry>

    其中:
        - x_qubits, y_qubits 均为 little-endian（q[0] 为最低位）
        - aux0 为结果寄存器（little-endian，长度必须等于 n，初态为 |0...0>）
        - aux1 为最高位进位位（初态为 |0>）

    说明:
        这里将 aux0 视为“结果寄存器”，而不是单个量子比特。
        因为两个 n 位数相加的结果需要 n+1 位来存储：
            低 n 位存入 aux0
            最高位进位存入 aux1

    参数:
        x_qubits : 第一个加数寄存器，little-endian
        y_qubits : 第二个加数寄存器，little-endian
        aux1     : 最高位进位位（单比特）
        aux0     : 结果寄存器，little-endian，长度为 n

    返回:
        QCircuit
    """
    cir = QCircuit()

    n = len(x_qubits)

    if len(y_qubits) != n:
        raise ValueError("要求 x_qubits 与 y_qubits 位数一致")

    if not hasattr(aux0, "__len__"):
        raise ValueError("这里 aux0 需要是 n 位结果寄存器，而不是单个量子比特")

    if len(aux0) != n:
        raise ValueError("要求 aux0 的位数与 x_qubits、y_qubits 一致")

    # QFT 内部统一采用 [MSB, ..., LSB] 的算术顺序：
    # full_res = [carry_msb, result_msb, ..., result_lsb]
    full_res = [aux1] + list(reversed(aux0))

    # 1) 先把结果寄存器送入傅里叶基
    cir << _qft_circuit(full_res, with_swap=False)

    # 2) 将 x 加到结果寄存器上
    for j, ctrl in enumerate(x_qubits):
        angles = _get_angles(1 << j, n + 1)
        for q, theta in zip(full_res, angles):
            if abs(theta) > 1e-15:
                cir << CR(ctrl, q, theta)

    # 3) 将 y 再加到结果寄存器上
    for j, ctrl in enumerate(y_qubits):
        angles = _get_angles(1 << j, n + 1)
        for q, theta in zip(full_res, angles):
            if abs(theta) > 1e-15:
                cir << CR(ctrl, q, theta)

    # 4) 逆 QFT，回到计算基
    cir << _iqft_circuit(full_res, with_swap=False)

    return cir



def QFTVarModAdd(x_qubits, y_qubits, aux1, aux0, N):
    """
    基于 QFT 的量子变量模加运算器（out-of-place）。

    实现:
        |x>|y>|0...0>|0>  ->  |x>|y>|(x+y) mod N>|flag>

    其中:
        - x_qubits, y_qubits 均为 little-endian（q[0] 为最低位）
        - aux0 为 n 位辅助结果寄存器，little-endian，初态需为 |0...0>
        - aux1 为 1 位辅助比特，初态需为 |0>

    构造思路:
        1) 先调用已有的 QFTAdder，得到 x+y 的 (n+1) 位和：
               aux1 : 最高位进位
               aux0 : 低 n 位结果
        2) 在完整寄存器 [aux1] + aux0 上减去常数 N，得到 |x+y-N>
        3) 若减法后 aux1=1，说明发生借位 / 需要加回 N，
           则对 aux0 条件性加 N，最终 aux0 中得到 (x+y) mod N

    注意:
        1) 本版本要求输入所编码的整数满足 0 <= x, y < N，
           这样只需做一次减 N 即可完成模约化。
        2) 本版本最终保证 aux0 中为正确的 (x+y) mod N，
           aux1 则保留“是否发生加回 N”这一判定位，不在本函数内清零。

    参数:
        x_qubits : 第一个加数寄存器，little-endian
        y_qubits : 第二个加数寄存器，little-endian
        aux1     : 1 位辅助判定位 / 进位位，初态为 |0>
        aux0     : n 位辅助结果寄存器，little-endian，初态为 |0...0>
        N        : 模数，要求 0 < N < 2^n

    返回:
        QCircuit
    """
    cir = QCircuit()

    n = len(x_qubits)

    if len(y_qubits) != n:
        raise ValueError("要求 x_qubits 与 y_qubits 位数一致")

    if not hasattr(aux0, "__len__"):
        raise ValueError("这里 aux0 需要是 n 位辅助结果寄存器，而不是单个量子比特")

    if len(aux0) != n:
        raise ValueError("要求 aux0 的位数与 x_qubits、y_qubits 一致")

    if N <= 0:
        raise ValueError("要求模数 N 为正整数")

    if N >= (1 << n):
        raise ValueError("要求模数 N 小于 2^n，寄存器位数不足")

    # 1) 先计算 x + y，结果写入辅助结果寄存器 aux0，最高位进位写入 aux1
    cir << QFTAdder(x_qubits, y_qubits, aux1, aux0)

    # 2) 构造完整和寄存器（QFT 内部使用 [MSB, ..., LSB]）
    #    full_sum = [aux1, aux0[n-1], ..., aux0[0]]
    full_sum = [aux1] + list(reversed(aux0))

    # 3) 计算 |x+y-N>
    cir << QFTConAdd(full_sum, -N)

    # 4) 若 aux1 = 1，说明需要对 |x+y-N> 加回 N
    #    这里最终只要求 aux0 中得到正确的模加结果
    cir << _ctrl_qft_con_add(list(reversed(aux0)), aux1, N)

    return cir



def QFTVarModDou(x_qubits, aux, N):
    """
    基于 QFT 的量子变量二倍模乘运算器（in-place）。

    实现:
        |x>|0>  ->  |(2x) mod N>|flag>

    其中:
        - x_qubits 为 little-endian（x_qubits[0] 为最低位）
        - aux 为 1 位辅助比特，初态需为 |0>

    构造思路:
        1) 先借助 SWAP 门完成“乘二倍”的左移操作：
              aux <- x 的最高位
              x   <- x 左移一位，最低位补 0
           即得到 full_reg = |2x> 的 (n+1) 位表示，其中
              full_reg = [aux] + x_qubits
        2) 在完整寄存器 [aux] + x_qubits 上减去常数 N，得到 |2x - N>
        3) 若减法后 aux = 1，说明 2x < N，需要对低 n 位 x_qubits 加回 N，
           从而使 x_qubits 中得到正确的 (2x) mod N

    注意:
        1) 本版本要求输入满足 0 <= x < N，且 N < 2^n
        2) 本版本最终保证 x_qubits 中为正确的 (2x) mod N
        3) aux 保留判定位信息，不在本函数内清零

    参数:
        x_qubits : 数据寄存器，little-endian
        aux      : 1 位辅助比特，初态为 |0>
        N        : 模数，要求 0 < N < 2^n

    返回:
        QCircuit
    """
    cir = QCircuit()

    n = len(x_qubits)

    if n == 0:
        raise ValueError("x_qubits 不能为空")
    if N <= 0:
        raise ValueError("要求模数 N 为正整数")
    if N >= (1 << n):
        raise ValueError("要求模数 N 小于 2^n，寄存器位数不足")

    cir << SWAP(aux, x_qubits[n - 1])
    for i in range(n - 1, 0, -1):
        cir << SWAP(x_qubits[i], x_qubits[i - 1])

    full_reg = [aux] + list(reversed(x_qubits))
    cir << QFTConAdd(full_reg, -N)

    cir << _ctrl_qft_con_add(list(reversed(x_qubits)), aux, N)

    return cir



def QFTVarModMul(x_qubits, y_qubits, z_qubits, work_msb, aux, N):
    """
    基于 QFT 的量子变量模乘运算器（out-of-place）。

    实现:
        |x>|y>|0>|0>|0...0>  ->  |x>|y>|0>|0>|(x*y) mod N>

    其中:
        - x_qubits, y_qubits, z_qubits 均为 little-endian（q[0] 为最低位）
        - z_qubits 为 n 位结果寄存器，初态需为 |0...0>
        - work_msb 为 1 位工作最高位 / 借位位，初态需为 |0>
        - aux 为 1 位辅助比特，初态需为 |0>

    比特总数:
        n + n + n + 1 + 1 = 3n + 2

    构造思路:
        由
            x = sum_i x_i 2^i
            y = sum_j y_j 2^j
        得
            x*y = sum_{i,j} x_i y_j 2^{i+j}

        因此在模 N 意义下：
            x*y mod N = sum_{i,j} x_i y_j (2^{i+j} mod N) mod N

        对每一对 (i, j)，当且仅当 x_i = 1 且 y_j = 1 时，
        就向 z_qubits 中加上常数 (2^(i+j) mod N)。
        这里使用已有的受控常数模加器 _ctrl_qft_con_mod_add，
        再对整段电路额外施加一层 control([x_i])，
        从而实现“双控制”的常数模加。

    参数:
        x_qubits : 第一个乘数寄存器，little-endian
        y_qubits : 第二个乘数寄存器，little-endian
        z_qubits : 结果寄存器，little-endian，长度为 n，初态为 |0...0>
        work_msb : 工作最高位 / 借位位（单比特），初态为 |0>
        aux      : 辅助位（单比特），初态为 |0>
        N        : 模数，要求 0 < N < 2^n

    返回:
        QCircuit
    """
    cir = QCircuit()

    n = len(x_qubits)

    if len(y_qubits) != n:
        raise ValueError("要求 x_qubits 与 y_qubits 位数一致")

    if len(z_qubits) != n:
        raise ValueError("要求 z_qubits 的位数与 x_qubits、y_qubits 一致")

    if N <= 0:
        raise ValueError("要求模数 N 为正整数")

    if N >= (1 << n):
        raise ValueError("要求模数 N 小于 2^n，寄存器位数不足")

    # 逐项累加 x_i * y_j * 2^(i+j) (mod N)
    for i, ctrl_x in enumerate(x_qubits):
        for j, ctrl_y in enumerate(y_qubits):
            add_const = (1 << (i + j)) % N

            # 加 0 为恒等操作，可直接跳过
            if add_const == 0:
                continue

            # 先构造“由 y_j 控制”的常数模加器，再整体受 x_i 控制，
            # 从而实现：
            #   当 x_i = 1 且 y_j = 1 时，
            #   z_qubits <- (z_qubits + 2^(i+j)) mod N
            add_cir = _ctrl_qft_con_mod_add(
                z_qubits,
                work_msb,
                aux,
                ctrl_y,
                add_const,
                N
            )

            cir << add_cir.control([ctrl_x])

    return cir



def QFTVarModSqr(x_qubits, z_qubits, work_msb, aux, N):
    """
    基于 QFT 的量子变量模平方运算器（out-of-place）。

    实现:
        |x>|0...0>|0>|0>  ->  |x>|x^2 mod N>|0>|0>

    其中:
        - x_qubits, z_qubits 均为 little-endian（q[0] 为最低位）
        - z_qubits 为 n 位结果寄存器，初态需为 |0...0>
        - work_msb 为 1 位工作最高位 / 借位位，初态需为 |0>
        - aux 为 1 位辅助比特，初态需为 |0>

    比特总数:
        n + n + 1 + 1 = 2n + 2

    构造思路:
        由
            x = sum_i x_i 2^i
        得
            x^2 = sum_i x_i 2^(2i) + 2 * sum_{i<j} x_i x_j 2^(i+j)

        因此在模 N 意义下：
            x^2 mod N
            = sum_i x_i (2^(2i) mod N)
              + sum_{i<j} x_i x_j (2^(i+j+1) mod N)   (mod N)

        所以：
        1) 对每个对角项 i，当 x_i = 1 时，向 z_qubits 中加上常数 2^(2i) mod N
        2) 对每个交叉项 i < j，当 x_i = x_j = 1 时，向 z_qubits 中加上常数 2^(i+j+1) mod N

        这里复用已有的受控常数模加器 _ctrl_qft_con_mod_add。
        对交叉项，先构造“由 x_j 控制”的模加器，再整体额外受 x_i 控制，
        从而实现双控制模加。

    参数:
        x_qubits : 输入寄存器，little-endian
        z_qubits : 结果寄存器，little-endian，长度为 n，初态为 |0...0>
        work_msb : 工作最高位 / 借位位（单比特），初态为 |0>
        aux      : 辅助位（单比特），初态为 |0>
        N        : 模数，要求 0 < N < 2^n

    返回:
        QCircuit
    """
    cir = QCircuit()

    n = len(x_qubits)

    if n == 0:
        raise ValueError("x_qubits 不能为空")

    if len(z_qubits) != n:
        raise ValueError("要求 z_qubits 的位数与 x_qubits 一致")

    if N <= 0:
        raise ValueError("要求模数 N 为正整数")

    if N >= (1 << n):
        raise ValueError("要求模数 N 小于 2^n，寄存器位数不足")

    for i, ctrl in enumerate(x_qubits):
        add_const = pow(2, 2 * i, N)

        if add_const == 0:
            continue

        cir << _ctrl_qft_con_mod_add(
            z_qubits,
            work_msb,
            aux,
            ctrl,
            add_const,
            N
        )

    for i in range(n):
        ctrl_i = x_qubits[i]

        for j in range(i + 1, n):
            ctrl_j = x_qubits[j]
            add_const = pow(2, i + j + 1, N)

            if add_const == 0:
                continue

            add_cir = _ctrl_qft_con_mod_add(
                z_qubits,
                work_msb,
                aux,
                ctrl_j,
                add_const,
                N
            )

            cir << add_cir.control([ctrl_i])

    return cir



def QFTPhaseEstimation(phase_qubits, sys_qubits, unitary_cir, state_prep=None):
    """
    基于 QFT 的量子相位估计运算器（QPE）。

    实现:
        |0>^{⊗t}|u>  ->  |phi_tilde>|u>

    其中:
        - phase_qubits 为 t 位相位寄存器
        - sys_qubits 为系统寄存器
        - U|u> = exp(2πi*phi)|u>
        - 对 phase_qubits 测量后，可得到相位 phi 的 t 位二进制近似

    参数:
        phase_qubits : 相位寄存器，采用算术顺序 [MSB, ..., LSB]
        sys_qubits   : 系统寄存器，可传单个量子比特或量子比特列表
        unitary_cir  : 作用在 sys_qubits 上的酉电路 U（QCircuit）
        state_prep   : 可选，系统寄存器初始化电路（QCircuit）

    返回:
        QCircuit
    """
    cir = QCircuit()

    # 统一 sys_qubits 的写法
    if hasattr(sys_qubits, "__len__") and not isinstance(sys_qubits, int):
        sys_qubits = list(sys_qubits)
    else:
        sys_qubits = [sys_qubits]

    phase_qubits = list(phase_qubits)
    t = len(phase_qubits)

    if t == 0:
        raise ValueError("phase_qubits 不能为空")

    if len(sys_qubits) == 0:
        raise ValueError("sys_qubits 不能为空")

    if unitary_cir is None:
        raise ValueError("unitary_cir 不能为空")

    if not hasattr(unitary_cir, "control"):
        raise ValueError("unitary_cir 需要是可被 .control([...]) 控制的 QCircuit")

    # 检查寄存器是否重叠
    overlap = set(phase_qubits) & set(sys_qubits)
    if overlap:
        raise ValueError(
            f"phase_qubits 与 sys_qubits 不能重叠，重复量子比特为: {sorted(list(overlap))}"
        )

    if state_prep is not None:
        cir << state_prep

    for q in phase_qubits:
        cir << H(q)

    for j, ctrl in enumerate(phase_qubits):
        repeat_times = 1 << j

        power_cir = QCircuit()
        for _ in range(repeat_times):
            power_cir << unitary_cir

        cir << power_cir.control([ctrl])

    cir << _iqft_circuit(phase_qubits, with_swap=False)

    return cir