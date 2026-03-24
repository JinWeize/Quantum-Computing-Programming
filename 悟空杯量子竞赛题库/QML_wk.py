"""
1：6-qubits 量子变分线路，四层变分线路，链式CNOT纠缠，RY门编码数值型变量，RZ门编码类型型变量
2：1层经典全连接层，将6-qubits输出的6位测量结果转化为predict_charges
3：损失函数采用RMSE，即（预测值-实际值）平方求和除以n再开根
4：优化器采用Adam优化器

数据预处理部分：
1：数值类型归一转化：(x-min)/(max-min) 再映射到[0, pi]上
2：类型变量转化为：是否吸烟1/0，地区：转化为数值编码？并利用旋转门RY或者RZ加载到量子比特上
3：对charges做对数化处理（可选）

参数计算：
1：数据加载6
2：变分层4*6*3(RY-YZ-RY)=72
3：经典层6+1=7
总计：79个参数

代码框架：
0. imports + config
1. 数据载入与预处理
2. 量子电路与测量
3. VQNet模型定义
4. 训练、验证、测试
5. 可视化与结果导出
6. main()
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pyqpanda3.core import CPUQVM, H, QCircuit, QProg, RY, measure, CNOT
from pyvqnet.data.data import data_generator
from pyvqnet.native.backprop_utils import AutoGradNode
from pyvqnet.nn import activation as F
from pyvqnet.nn.linear import Linear
from pyvqnet.nn.module import Module
from pyvqnet.optim.adam import Adam
from pyvqnet.tensor import QTensor, tensor

def data_load(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    读取训练集数据以及测试集数据。
    :param train_path: 训练集 csv 路径
    :param test_path:测试集 csv 路径
    :return:数据元组
    """


def split_train_eval(
        train_df: pd.DataFrame,
        val_ratio: float = 0.2,
        seed: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """

        :param train_df:
        :param val_ratio:
        :param seed:
        :return:
        """


def data_processor(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_ratio: float = 0.2,
        use_log_target: bool = True,
        seed: int = 42
    ) -> dict:
        """
        对数据进行预处理。
        :param train_df:
        :param test_df:
        :param val_ratio:
        :param use_log_target:bool
        :param seed:
        :return:预处理后的数据，以字典形式给出。
        """


def inverse_target_transform(y_pred_proc: np.ndarray, meta: dict) -> np.ndarray:
    """
    对数化处理逆变换。
    :param y_pred_proc:预测得到的对数化医疗保险费用预测值
    :param meta:包含预处理后数据的字典
    :return:预测值正常医疗保险费用
    """


def make_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """

    :param X: 九维输入数据
    :param y: 预测医疗保险值
    :param batch_size: batch的大小
    :param shuffle: 是否打散数据，随机取样
    :return: 训练用batch
    """


def init_circuit(n_qubits: int = 6) -> tuple[object, list[int]]:
    """
    初始化量子电路
    :param n_qubits: 量子比特数
    :return:
    """
