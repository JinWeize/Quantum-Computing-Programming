import math
import numpy as np
import QFTFunction

from pyqpanda3.core import QCircuit, H, X, CR

# 通常取n = log(2, N),向上取整。t = 2 * n

def Shor(phase_qubits, acc_qubits, tmp_qubits, work_msb, aux, a, N):
    """
    基于 QFT 的 Shor 周期求解电路（量子部分）。

    构造:
        |0>^{⊗t} |1> |0...0>
            --H on phase register--
            --controlled modular exponentiation--
            --IQFT on phase register-->
        最终返回整个量子线路 cir

    参数
    ----
    phase_qubits : list
        第一寄存器（相位寄存器 / 指数寄存器），little-endian
        即 phase_qubits[0] 为最低位，对应 2^0
    acc_qubits : list
        第二寄存器（累乘 / 结果寄存器），little-endian
        本函数内部会将其初始化到 |1>
    tmp_qubits : list
        模乘辅助寄存器，little-endian，长度需与 acc_qubits 一致，初态应为 |0...0>
    work_msb : int
        QFT 模加 / 模乘内部使用的工作位，初态应为 |0>
    aux : int
        QFT 模加 / 模乘内部使用的辅助位，初态应为 |0>
    a : int
        底数，要求 gcd(a, N) = 1
    N : int
        待分解整数中的模数

    返回
    ----
    cir : QCircuit

    说明
    ----
    1) phase_qubits 这里采用 little-endian，是为了与
       QFTFunction.QFTConModExp 的接口保持一致。
    2) 本函数返回的只是“量子线路”，不包含测量。
    3) 若你后续只测量 phase_qubits，并希望 CFD 直接按通常二进制串解析，
       建议测量时把“最高位量子比特 -> 最高位经典比特”，
       即让打印出来的 bitstring 是 MSB -> LSB 的阅读顺序。
    """

    def _iqft_arith(qubits_msb_to_lsb):
        """
        局部逆 QFT。
        输入 qubits 必须按 [MSB, ..., LSB] 顺序传入。
        """
        iqft_cir = QCircuit()
        n = len(qubits_msb_to_lsb)

        for i in range(n - 1, -1, -1):
            target = qubits_msb_to_lsb[i]
            for j in range(n - 1, i, -1):
                control = qubits_msb_to_lsb[j]
                angle = -math.pi / (2 ** (j - i))
                iqft_cir << CR(control, target, angle)
            iqft_cir << H(target)

        return iqft_cir

    cir = QCircuit()

    phase_qubits = list(phase_qubits)
    acc_qubits = list(acc_qubits)
    tmp_qubits = list(tmp_qubits)

    if len(phase_qubits) == 0:
        raise ValueError("phase_qubits 不能为空")
    if len(acc_qubits) == 0:
        raise ValueError("acc_qubits 不能为空")
    if len(tmp_qubits) != len(acc_qubits):
        raise ValueError("要求 tmp_qubits 与 acc_qubits 位数一致")
    if N <= 1:
        raise ValueError("要求 N > 1")
    if N >= (1 << len(acc_qubits)):
        raise ValueError("要求 N < 2^n，其中 n = len(acc_qubits)")
    if math.gcd(a, N) != 1:
        raise ValueError("要求 gcd(a, N) = 1")

    # 1) 初始化第二寄存器到 |1>
    # little-endian 下，整数 1 对应最低位为 1
    cir << X(acc_qubits[0])

    # 2) 对第一寄存器做 H，制备均匀叠加
    for q in phase_qubits:
        cir << H(q)

    # 3) 受控模指数运算
    #    复用你上传文件中的 QFTConModExp
    cir << QFTFunction.QFTConModExp(
        phase_qubits,
        acc_qubits,
        tmp_qubits,
        work_msb,
        aux,
        a,
        N
    )

    # 4) 对第一寄存器做逆 QFT
    #    由于局部 IQFT 例程按 [MSB, ..., LSB] 工作，
    #    而 phase_qubits 采用 little-endian，因此这里取 reversed
    cir << _iqft_arith(list(reversed(phase_qubits)))

    return cir


def CFD(meas_result, a, N, t=None, bit_order="msb", top_k=8):
    """
    根据 Shor 第一寄存器的测量结果，使用连分数法恢复周期 r。
    参数
    ----
    meas_result : dict / str / int
        1) 若为 dict，通常就是 qvm.result().get_counts() 的返回值；
           这里默认该 counts 只包含“第一寄存器”的测量结果。
        2) 若为 str，则应是一个二进制串，如 '010110'
        3) 若为 int，则应是测得的整数 c，此时必须额外给出 t
    a : int
        Shor 中选取的底数
    N : int
        模数
    t : int or None
        第一寄存器位数。若 meas_result 为 str / dict，通常可自动推断；
        若 meas_result 为 int，则必须显式提供 t
    bit_order : str
        "msb"    : 二进制串按 MSB -> LSB 阅读（默认）
        "little" : 二进制串按 LSB -> MSB 阅读，此时内部会自动翻转
    top_k : int
        当 meas_result 是 dict 时，按出现次数从高到低尝试前 top_k 个峰

    返回
    ----
    r : int or None
        恢复出的周期。若未成功恢复，则返回 None

    说明
    ----
    1) 本函数会先对 c / 2^t 做连分数展开，得到一系列收敛分数 p/q。
    2) 若 q 只是 r 的因子，本函数还会尝试 q 的若干倍数。
    3) 一旦找到满足 a^r ≡ 1 (mod N) 的最小正整数，就返回它。
    """

    def _normalize_bitstring(bits):
        bits = str(bits).replace(" ", "").replace("_", "")
        if bits.startswith("0b"):
            bits = bits[2:]
        if len(bits) == 0:
            raise ValueError("空二进制串")
        if any(ch not in "01" for ch in bits):
            raise ValueError(f"非法二进制串: {bits}")
        return bits

    def _continued_fraction(num, den):
        """
        对有理数 num / den 做连分数展开。
        返回系数列表 [a0, a1, ...]
        """
        cf = []
        while den != 0:
            a0 = num // den
            cf.append(a0)
            num, den = den, num - a0 * den
        return cf

    def _convergents(cf):
        """
        根据连分数系数，生成所有收敛分数 (p, q)
        """
        convs = []
        p_nm2, p_nm1 = 0, 1
        q_nm2, q_nm1 = 1, 0

        for ai in cf:
            p_n = ai * p_nm1 + p_nm2
            q_n = ai * q_nm1 + q_nm2
            convs.append((p_n, q_n))
            p_nm2, p_nm1 = p_nm1, p_n
            q_nm2, q_nm1 = q_nm1, q_n

        return convs

    def _reduce_to_minimal_order(r_candidate):
        """
        已知 a^(r_candidate) ≡ 1 (mod N)，继续约化到最小正周期
        """
        r = r_candidate
        changed = True

        while changed:
            changed = False
            d = 2
            while d * d <= r:
                if r % d == 0 and pow(a, r // d, N) == 1:
                    r //= d
                    changed = True
                    break
                d += 1

        return r

    def _try_recover_from_c(c, t_local):
        """
        由单个测量值 c 恢复周期。
        """
        if t_local <= 0:
            raise ValueError("t 必须为正整数")

        Q = 1 << t_local

        if c <= 0 or c >= Q:
            return None

        # 对 c / Q 做连分数展开
        cf = _continued_fraction(c, Q)
        convs = _convergents(cf)

        # 逐个尝试收敛分母 q
        for _, q in convs:
            if q <= 0:
                continue
            if q >= N:
                break

            # 先试 q 自己，再试 q 的若干倍数
            max_k = max(1, N // q)
            for k in range(1, max_k + 1):
                cand = k * q
                if cand <= 0:
                    continue

                if pow(a, cand, N) == 1:
                    return _reduce_to_minimal_order(cand)

        return None

    if N <= 1:
        raise ValueError("要求 N > 1")
    if math.gcd(a, N) != 1:
        raise ValueError("要求 gcd(a, N) = 1")

    # -------- 情况 1：counts 字典 --------
    if isinstance(meas_result, dict):
        if len(meas_result) == 0:
            return None

        keys = list(meas_result.keys())
        vals = np.array(list(meas_result.values()), dtype=int)
        order = np.argsort(-vals)

        tried = 0
        for idx in order:
            if tried >= top_k:
                break

            bits = _normalize_bitstring(keys[int(idx)])
            if t is None:
                t_local = len(bits)
            else:
                t_local = t
                if len(bits) != t_local:
                    raise ValueError("counts 中 bitstring 长度与参数 t 不一致")

            if bit_order.lower() in ("little", "le", "lsb"):
                bits = bits[::-1]
            elif bit_order.lower() not in ("msb", "be", "big"):
                raise ValueError("bit_order 只支持 'msb' 或 'little'")

            c = int(bits, 2)
            r = _try_recover_from_c(c, t_local)
            if r is not None:
                return r

            tried += 1

        return None

    # -------- 情况 2：单个 bitstring --------
    if isinstance(meas_result, str):
        bits = _normalize_bitstring(meas_result)
        t_local = len(bits) if t is None else t

        if len(bits) != t_local:
            raise ValueError("bitstring 长度与参数 t 不一致")

        if bit_order.lower() in ("little", "le", "lsb"):
            bits = bits[::-1]
        elif bit_order.lower() not in ("msb", "be", "big"):
            raise ValueError("bit_order 只支持 'msb' 或 'little'")

        c = int(bits, 2)
        return _try_recover_from_c(c, t_local)

    # -------- 情况 3：单个整数 c --------
    if isinstance(meas_result, int):
        if t is None:
            raise ValueError("当 meas_result 为整数时，必须显式给出 t")
        return _try_recover_from_c(meas_result, t)

    raise TypeError("meas_result 只支持 dict / str / int")