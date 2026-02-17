from pyqpanda3.core import QCircuit, CNOT, SWAP, X, Y, Z, S, H, T, QProg
from pyqpanda3.quantum_info import Unitary
import numpy as np

def R_a(alpha):
    return np.array([[np.cos(alpha),np.sin(alpha)],[np.sin(-1 * alpha), np.cos(alpha)]])

def T_b(beta):
    return np.array([[np.exp(1j * beta), 0],[0, np.exp(-1j * beta)]])


print(-1j * np.dot(np.dot(T_b(np.pi / 2), R_a(np.pi / 2)), T_b(0)))
print(-1j * np.dot(np.dot(T_b(np.pi / 2), R_a(np.pi / 2)), T_b(0)))
print(-1j * np.dot(np.dot(T_b(np.pi / 2), R_a(np.pi / 4)), T_b(0)))
print(-1j * np.dot(np.dot(T_b(np.pi / 2), R_a(np.pi / 4)), T_b(0)))