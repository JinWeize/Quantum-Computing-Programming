"""
1：6-qubits 量子变分线路，四层变分线路，链式CNOT纠缠，RY门编码数值型变量，RZ门编码类型型变量
2：1层经典全连接层，将6-qubits输出的6位测量结果转化为predict_charges
3：损失函数采用RMSE，即（预测值-实际值）平方求和除以n再开根
4：优化器采用Adam优化器

数据预处理部分：
1：数值类型归一转化：(x-min)/(max-min) 再映射到[0, pi]上
2：类型变量转化为：是否吸烟1/0，地区：转化为独热编码？并利用旋转门RY或者RZ加载到量子比特上
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
import pandas as pd
from sklearn.model_selection import train_test_split
import math

from pyqpanda3.core import CPUQVM, H, QCircuit, QProg, RY, RZ,measure, CNOT
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
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if train_df.shape[1] != 7 or test_df.shape[1] != 7:
        raise ValueError("缺少charges")
    if train_df.isnull().any().any() or test_df.isnull().any().any():
        raise ValueError("存在缺失值")

    expected_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']

    train_df = train_df[expected_cols]
    test_df = test_df[expected_cols]

    return train_df, test_df


def data_min_max(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    数据预处理辅助函数
    :param data: 输入pd.df文件
    :return: 对列做min-max处理后输出
    """
    min = data[col].min()
    max = data[col].max()

    data[col] = math.pi * (data[col] - min) / (max - min)
    return data


def data_region(data: pd.DataFrame, region_order) -> pd.DataFrame:
    region_dummies = pd.get_dummies(data['region'], columns=region_order)
    region_dummies = region_dummies[region_order]

    data = pd.concat([data, region_dummies], axis=1)
    data = data.drop('region', axis=1)

    return data


def data_processor(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_ratio: float = 0.2,
        use_log_target: bool = True,
        seed: int = 42
    ) -> dict:
        """
        对数据进行预处理。
        :param train_df:训练数据
        :param test_df:测试数据
        :param val_ratio:
        :param use_log_target:bool，是否进行对数化处理
        :param seed:
        :return:预处理后的数据，以字典形式给出。
        """
        # 数据分类和标签
        X1 = train_df.drop('charges', axis=1)
        y1 = train_df['charges']

        X_train, X_val, y_train, y_val = train_test_split(X1, y1, test_size=val_ratio,
                                                          random_state = seed)
        X_test = test_df.drop('charges', axis=1)
        y_test = test_df['charges']
        # print(y_train)
        sex_map = {'male': 1.0, 'female': 0.0}
        smoker_map = {'yes': 1.0, 'no': 0.0}
        region_order = ['northeast', 'northwest', 'southeast', 'southwest']
        col_order = ['age', 'bmi', 'children', 'sex', 'smoker',
                     'northeast', 'northwest', 'southeast', 'southwest'
                     ]

        # 标准化数据
        X_train = data_min_max(X_train, 'age')
        X_train = data_min_max(X_train, 'bmi')
        X_train = data_min_max(X_train, 'children')
        X_val = data_min_max(X_val, 'age')
        X_val = data_min_max(X_val, 'bmi')
        X_val = data_min_max(X_val, 'children')
        X_test = data_min_max(X_test, 'age')
        X_test = data_min_max(X_test, 'bmi')
        X_test = data_min_max(X_test, 'children')

        X_train['sex'] = X_train['sex'].map(sex_map)
        X_train['smoker'] = X_train['smoker'].map(smoker_map)
        X_val['sex'] = X_val['sex'].map(sex_map)
        X_val['smoker'] = X_val['smoker'].map(smoker_map)
        X_test['sex'] = X_test['sex'].map(sex_map)
        X_test['smoker'] = X_test['smoker'].map(smoker_map)

        X_train = data_region(X_train, region_order)
        X_val = data_region(X_val, region_order)
        X_test = data_region(X_test, region_order)

        X_train = X_train[col_order]
        X_val = X_val[col_order]
        X_test = X_test[col_order]

        if use_log_target:
            y_train = np.log1p(y_train)
            y_val = np.log1p(y_val)
            y_test = np.log1p(y_test)
        # print(y_train['charges'])
        N_train = X_train.shape[0]
        N_val = X_val.shape[0]
        N_test = X_test.shape[0]
        # print(N_train)
        X_train = np.array(X_train, dtype = np.float32).reshape(N_train, 9)
        X_val = np.array(X_val, dtype=np.float32).reshape(N_val, 9)
        X_test = np.array(X_test, dtype=np.float32).reshape(N_test, 9)

        y_train = y_train.to_numpy(dtype = np.float32).reshape(N_train, 1)
        y_val = y_val.to_numpy(dtype=np.float32).reshape(N_val, 1)
        y_test = y_test.to_numpy(dtype=np.float32).reshape(N_test, 1)

        meta = {'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test,
                'region_order': region_order,
                'data_order': col_order
        }

        return meta


def init_circuit(n_qubits: int = 6) -> tuple[object, list[int]]:
    """
    初始化量子电路
    :param n_qubits: 量子比特数
    :return:
    """
    qvm = CPUQVM()
    qubits = list(range(n_qubits))

    return qvm, qubits


def circuit(x: np.ndarray, theta: np.ndarray, qvm, qubits: list[int],
            n_layers: int = 4, shots:int = 2048) -> np.ndarray:
    """
    创建量子电路，并利用量子虚拟机进行测量和统计，得到测量期望值并返回。
    :param x: 单个样本
    :param theta: 角度编码shape(4, 6, 3)
    :param qvm: 量子虚拟机
    :param qubits: 量子比特
    :param n_layers: 变分层层数
    :param shots: 测量次数
    :return: 对六个量子比特的测量期望值
    """
    prog = QProg()
    cir = QCircuit()
    age, bmi, children, sex, smoker, r_ne, r_nw, r_se, r_sw = x
    region_vec = [r_ne, r_nw, r_se, r_sw]

    for layer in range(n_layers):
        # 编码层
        cir << RY(qubits[0], age)
        cir << RY(qubits[1], bmi)
        cir << RY(qubits[2], children)
        cir << RZ(qubits[3], math.pi * sex)
        cir << RZ(qubits[4], math.pi * smoker)
        cir << RY(qubits[5], math.pi * region_vec[layer])
        # 变分层
        for i in range(6):
            cir << RY(qubits[i], float(theta[layer, i, 0]))
            cir << RZ(qubits[i], float(theta[layer, i, 1]))
            cir << RY(qubits[i], float(theta[layer, i, 2]))
        # 纠缠层
        cir << CNOT(qubits[0], qubits[1])
        cir << CNOT(qubits[1], qubits[2])
        cir << CNOT(qubits[2], qubits[3])
        cir << CNOT(qubits[3], qubits[4])
        cir << CNOT(qubits[4], qubits[5])

    prog << cir
    for i in range(6):
        prog << measure(qubits[i], i)

    qvm.run(prog, shots)
    counts = qvm.result().get_counts()

    z_exp = np.zeros(6, dtype=np.float32)
    for bitstr, cnt in counts.items():
        for i, b in enumerate(bitstr):
            z_exp[i] += (1.0 if b == '0' else -1.0) * cnt

    z_exp /= float(shots)
    return z_exp


class Hybrid(Module):
    """创建量子层，定义前向函数"""
    def __init__(self, n_qubits: int = 6, n_layers: int = 4,
                 shots: int = 2048, shift: float = math.pi / 2):
        super(Hybrid, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.shift = shift
        self.qvm, self.qubits = init_circuit(n_qubits)
        theta_init  = 0.01 * np.random.randn(n_layers, n_qubits, 3).astype(np.float32)
        self.theta = QTensor(theta_init, requires_grad=True)

    def forward(self, x_batch: QTensor) -> QTensor:
        x_np = np.array(x_batch.data, dtype=np.float32)  #(B, 9)
        theta_np = np.array(self.theta.data, dtype=np.float32)

        q_out = []
        for x in x_np:
            q_feat = circuit(x,theta=theta_np,
                              qvm=self.qvm, qubits=self.qubits,
                              n_layers=self.n_layers, shots=self.shots)
            q_out.append(q_feat)
        q_out = np.array(q_out, dtype=np.float32)
        requires_grad = x_batch.requires_grad or self.theta.requires_grad


        def _backward_theta(g, x_np):
            g_np = np.array(g, dtype=np.float32)
            theta_np = np.array(self.theta.data, dtype=np.float32)

            grad_theta = np.zeros_like(theta_np, dtype=np.float32)

            for l in range(self.n_layers):
                for q in range(self.n_qubits):
                    for k in range(3):
                        theta_plus = theta_np.copy()
                        theta_minus = theta_np.copy()

                        theta_plus[l, q, k] += self.shift
                        theta_minus[l, q, k] -=self.shift

                        batch_plus = []
                        batch_minus = []

                        for x in x_np:
                            out_plus = circuit(
                                x=x,theta=theta_plus,
                                qvm=self.qvm,qubits=self.qubits,
                                n_layers=self.n_layers,shots=self.shots
                            )
                            out_minus = circuit(
                                x=x, theta=theta_minus,
                                qvm=self.qvm, qubits=self.qubits,
                                n_layers=self.n_layers, shots=self.shots
                            )
                            batch_plus.append(out_plus)
                            batch_minus.append(out_minus)
                        batch_plus = np.array(batch_plus, dtype=np.float32)
                        batch_minus = np.array(batch_minus, dtype=np.float32)

                        local_grad = (batch_plus - batch_minus) / 2
                        grad_theta[l, q, k] = np.sum(local_grad * g_np)
            return grad_theta
        nodes = []
        if self.theta.requires_grad:
            nodes.append(AutoGradNode(
                tensor = self.theta, df=lambda g: _backward_theta(g, x_np)
            )
            )

        return QTensor(data=q_out, requires_grad=requires_grad,nodes=nodes)


class Net(Module):
    """创建VQNet层，整合量子层与经典全连接层,4层变分量子层+1层全连接层"""
    def __init__(self, n_qubits: int = 6, n_layers: int = 4, shots: int = 2048):
        super(Net, self).__init__()
        self.hybrid = Hybrid(n_qubits=n_qubits,
                             n_layers=n_layers,
                             shots=shots,
                             )
        self.fc = Linear(6, 1)

    def forward(self, x_batch: QTensor) -> QTensor:
        q_feat = self.hybrid(x_batch)
        y_pred = self.fc(q_feat)

        return y_pred

def train_and_eval(data_dict: dict, config: dict) -> dict:
    """
    训练以及测试。
    :param data_dict:data_processor的输出
    :param config: 参数字典
    :return: 训练结果字典
    """
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_val = data_dict["X_val"]
    y_val = data_dict["y_val"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]

    n_qubits = config.get("n_qubits", 6)
    n_layers = config.get("n_layers", 4)
    shots = config.get("shots", 256)
    epochs = config.get("epochs", 20)
    batch_size = config.get("batch_size", 4)
    lr = config.get("lr", 1e-2)
    use_log_target = config.get("use_log_target", True)
    eps = config.get("eps", 1e-8)
    verbose = config.get("verbose", True)

    model = Net(
        n_qubits=n_qubits,
        n_layers=n_layers,
        shots=shots
    )
    optimizer = Adam(model.parameters(), lr=lr)

    def rmse_loss_qtensor(y_pred: QTensor, y_true: QTensor) -> QTensor:
        diff = y_pred - y_true
        mse = tensor.mean(tensor.square(diff))
        rmse = tensor.sqrt(mse + eps)
        return rmse

    def evaluate_numpy(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        model.eval()

        pred_list = []
        for xb, _ in data_generator(X, y, batch_size=1, shuffle=False):
            xb = QTensor(np.array(xb, dtype=np.float32), requires_grad=False)
            out = model(xb)
            pred_np = np.array(out.data, dtype=np.float32).reshape(-1, 1)
            pred_list.append(pred_np)

        y_pred = np.vstack(pred_list).astype(np.float32)
        y_true = np.array(y, dtype=np.float32).reshape(-1, 1)

        if use_log_target:
            y_pred_eval = np.expm1(y_pred)
            y_true_eval = np.expm1(y_true)
        else:
            y_pred_eval = y_pred
            y_true_eval = y_true

        rmse = float(np.sqrt(np.mean((y_pred_eval - y_true_eval) ** 2)))
        return rmse, y_pred_eval

    history = {
        "train_loss": [],
        "val_rmse": [],
        "test_rmse": []
    }

    best_val_rmse = float("inf")
    best_test_rmse = None
    best_test_pred = None

    for epoch in range(1, epochs + 1):
        model.train()
        batch_loss_list = []

        for xb, yb in data_generator(X_train, y_train, batch_size=batch_size, shuffle=True):
            xb = QTensor(np.array(xb, dtype=np.float32), requires_grad=False)
            yb = QTensor(np.array(yb, dtype=np.float32), requires_grad=False)

            optimizer.zero_grad()

            y_pred = model(xb)
            loss = rmse_loss_qtensor(y_pred, yb)

            loss.backward()
            optimizer._step()

            loss_value = float(np.array(loss.data).reshape(-1)[0])
            batch_loss_list.append(loss_value)

        train_loss = float(np.mean(batch_loss_list)) if len(batch_loss_list) > 0 else np.nan
        val_rmse, _ = evaluate_numpy(X_val, y_val)
        test_rmse, test_pred = evaluate_numpy(X_test, y_test)

        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_rmse)
        history["test_rmse"].append(test_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_test_rmse = test_rmse
            best_test_pred = test_pred

        if verbose:
            print(
                f"Epoch [{epoch}/{epochs}] | "
                f"train_loss = {train_loss:.6f} | "
                f"val_rmse = {val_rmse:.6f} | "
                f"test_rmse = {test_rmse:.6f}"
            )

    return {
        "model": model,
        "history": history,
        "best_val_rmse": best_val_rmse,
        "best_test_rmse": best_test_rmse,
        "test_pred": best_test_pred
    }

def plot_train_curves(history: dict, save_path: str | None) -> None:
    """
    可视化函数，绘制训练过程中的损失函数损失值曲线。
    :param history: 训练数据，训练损失值与测试RMSE值
    :param save_path: 图片保存路径
    :return: figure
    """
    train_loss = history['train_loss']
    val_rmse = history["val_rmse"]
    test_rmse = history["test_rmse"]
    xrange = range(1, len(train_loss) + 1)

    ax1 = plt.subplot()
    plt.plot()

def main() -> None:
    train_path = "D:/学校文件/2025天目启航/本源量子悟空杯/train_wk.csv"
    test_path = "D:/学校文件/2025天目启航/本源量子悟空杯/test_wk.csv"
    train_df, test_df = data_load(train_path, test_path)

    # print(data_processor(train_df, test_df))

    meta = data_processor(train_df, test_df)
    config = {
        "n_qubits": 6,
        "n_layers": 4,
        "shots": 256,
        "epochs": 100,
        "batch_size": 64,
        "lr": 5e-3,
        "use_log_target": True,
        "verbose": True
    }
    results = train_and_eval(meta, config)

    print("best_val_rmse =", results["best_val_rmse"])
    print("best_test_rmse =", results["best_test_rmse"])


if __name__ == "__main__":
    main()