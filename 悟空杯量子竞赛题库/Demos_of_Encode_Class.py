from pyqpanda3.core import Encode, QProg, QCircuit, CPUQVM
from pyqpanda3.core.core import measure
import numpy as np
import math

"""
振幅编码amplitude_encode
递归振幅编码amplitude_encode_recursive
角度编码angle_encode (qubits, angle)
稠密角度编码dense_angle_encode
将字符串数据映射到量子比特上basic_encode( str )
获得保真度Fidelity  get_fidelity
"""
def basic_encode_demo():
    qvm = CPUQVM()

    x = '1001'
    qubits = [0, 1, 2, 3]

    cir_encode = Encode()

    cir_encode.basic_encode(qubits, x)

    prog = QProg()
    prog << cir_encode.get_circuit()
    for i in qubits:
        prog << measure(qubits[i], i)

    encode_qubits = cir_encode.get_out_qubits()

    qvm.run(prog, 1000)
    result = qvm.result().get_counts()

    return result



def angle_encode_demo():
    qvm = CPUQVM()
    prog = QProg()
    x = [np.pi, np.pi]
    qubits = [0, 1]

    cir_encode = Encode()

    cir_encode.angle_encode(qubits, x)
    prog << cir_encode.get_circuit()

    prog <<measure(qubits[0], 0) \
        <<measure(qubits[1], 1)

    qvm.run(prog, 1000)
    result = qvm.result().get_counts()

    return result



def dense_angle_encode_demo():
    qvm = CPUQVM()
    prog = QProg()
    x = [np.pi, np.pi]
    qubits = [0, 1]

    cir_encode = Encode()

    cir_encode.dense_angle_encode(qubits, x)
    prog << cir_encode.get_circuit()

    qvm.run(prog, 1)
    result = qvm.result().get_state_vector()

    return result


# TOP-DOWN编码amplitude_encode
def amplitude_encode_demo():
    qvm = CPUQVM()
    prog = QProg()

    qubits = [0, 1, 2]  # 传入的数据数量需要满足2^len(qubits)
    x = [(6 + 1j) / math.sqrt(37), 0j, 0j, 0j, 0j, 0j, 0j, 0j]  # 传入复数版本时，需要对数据进行归一化处理再传入

    cir_encode = Encode()
    cir_encode.amplitude_encode(qubits, x)
    prog << cir_encode.get_circuit()

    qvm.run(prog, 1)
    state = qvm.result().get_state_vector()

    return state


# BOTTOM-TOP振幅编码dc_amplitude_encode
def dc_amplitude_encode_demo():
    qvm = CPUQVM()
    prog = QProg()

    qubits = [0, 1, 2]  # 传入的数据数量需要满足len(qubits)
    x = [1, 0.0, 0.0]  # 只支持float形式的数据

    cir_encode = Encode()
    cir_encode.dc_amplitude_encode(qubits, x)
    prog << cir_encode.get_circuit()

    qvm.run(prog, 1)
    state = qvm.result().get_state_vector()

    return state


# 双向振幅编码bid_amplitude_encode
def bid_amplitude_encode_demo():
    qvm = CPUQVM()
    prog = QProg()

    qubits = [0, 1, 2]  # 线路宽度见书,尽量大于log(len(data), 2)
    x = [1 / math.sqrt(2), 0.0, 0.0, 1 / math.sqrt(2)]  # 只支持float形式的数据

    cir_encode = Encode()
    cir_encode.bid_amplitude_encode(qubits, x)
    prog << cir_encode.get_circuit()

    qvm.run(prog, 1)
    state = qvm.result().get_state_vector()

    return state


# 基于Schmidt分解的振幅编码
def schmidt_enncode_demo():
    qvm = CPUQVM()
    prog = QProg()

    qubits = [0, 1, 2]  # 线路宽度见书,尽量大于log(len(data), 2)
    x = [0, 1 / math.sqrt(3), 0, 0, 0, 1 / math.sqrt(3), 1 / math.sqrt(3), 0]  # 只支持float形式的数据

    cir_encode = Encode()
    cir_encode.schmidt_encode(qubits, x, 0) # 0为截断小奇异值的阈值
    prog << cir_encode.get_circuit()

    qvm.run(prog, 1)
    state = qvm.result().get_state_vector()

    return state

print("字符串编码示例：x=‘1001’", basic_encode_demo())
print("角度编码，pi，pi：", angle_encode_demo())
print("稠密角度编码，pi，pi：", dense_angle_encode_demo())
print("Top-down振幅编码x=6 + 1j：", amplitude_encode_demo())
print("Bottom-Top振幅编码x=1：", dc_amplitude_encode_demo())
print("bid_amplitude_encode双向振幅编码x=[1 / math.sqrt(2), 0.0, 0.0, 1 / math.sqrt(2)]：", bid_amplitude_encode_demo())
print("schmidt编码x=[0, 1 / math.sqrt(3), 0, 0, 0, 1 / math.sqrt(3), 1 / math.sqrt(3), 0]：", schmidt_enncode_demo())