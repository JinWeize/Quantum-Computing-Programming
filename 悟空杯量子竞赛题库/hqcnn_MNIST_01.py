"""
Hybrid Quantum-Classical CNN (HQCNN) for MNIST 0/1 classification.

This script is a pyqpanda3 adaptation of the code structure shown in the
user-provided PDF. The overall logic is kept the same:
1. Download and load MNIST.
2. Keep only labels 0 and 1.
3. Build a one-qubit quantum layer driven by a classical network output.
4. Wrap that quantum layer into a VQNet Module with manual backward logic.
5. Train / evaluate / visualize.

Notes
-----
- The original PDF uses pyqpanda / pyQPanda2 style APIs.
- This file replaces only the quantum-circuit part with pyqpanda3 APIs.
- The VQNet-side custom autograd node is written using the current public
  tutorial style (`AutoGradNode`), which is closer to recent VQNet examples.
- I kept the quantum output consistent with the original sample code:
  it returns the sampled mean of the measurement bit on one qubit.
  If you want the strict Pauli-Z expectation, replace the return value with
  `1.0 - 2.0 * p1`.

Dependencies
------------
pip install pyqpanda3 pyvqnet numpy matplotlib
"""

from __future__ import annotations

import gzip
import os
import struct
from array import array as pyarray
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pyqpanda3.core import CPUQVM, H, QCircuit, QProg, RY, measure
from pyvqnet.data.data import data_generator
from pyvqnet.native.backprop_utils import AutoGradNode
from pyvqnet.nn import activation as F
from pyvqnet.nn.conv import Conv2D
from pyvqnet.nn.linear import Linear
from pyvqnet.nn.loss import CategoricalCrossEntropy
from pyvqnet.nn.module import Module
from pyvqnet.nn.pooling import MaxPool2D
from pyvqnet.optim.adam import Adam
from pyvqnet.tensor import QTensor, tensor

try:
    matplotlib.use("TkAgg")
except Exception:
    print("Can not use matplot TkAgg")

try:
    import urllib.request
except ImportError as exc:
    raise ImportError("You should use Python 3.x") from exc


# ---------------------------------------------------------------------------
# 1) Data preparation
# ---------------------------------------------------------------------------
# The old PDF uses Yann LeCun's URL. The current VQNet tutorial uses the OSSCI
# mirror below, which is usually more stable.
URL_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
KEY_FILE: Dict[str, str] = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz",
}


def _download(dataset_dir: str, file_name: str) -> None:
    """Download one MNIST file and unzip it if needed."""
    os.makedirs(dataset_dir, exist_ok=True)
    file_path = os.path.join(dataset_dir, file_name)

    if os.path.exists(file_path):
        with gzip.GzipFile(file_path) as f:
            file_path_ungz = file_path[:-3].replace("\\", "/")
            if not os.path.exists(file_path_ungz):
                with open(file_path_ungz, "wb") as out_f:
                    out_f.write(f.read())
        return

    print(f"Downloading {file_name} ...")
    urllib.request.urlretrieve(URL_BASE + file_name, file_path)

    if os.path.exists(file_path):
        with gzip.GzipFile(file_path) as f:
            file_path_ungz = file_path[:-3].replace("\\", "/")
            file_path_ungz = file_path_ungz.replace("-idx", ".idx")
            if not os.path.exists(file_path_ungz):
                with open(file_path_ungz, "wb") as out_f:
                    out_f.write(f.read())
    print("Done")



def download_mnist(dataset_dir: str) -> None:
    """Download all MNIST gzip files."""
    for v in KEY_FILE.values():
        _download(dataset_dir, v)



def load_mnist(
    dataset: str = "training_data",
    digits: np.ndarray = np.arange(2),
    path: str = "./mnist_data",
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST images and labels into numpy arrays."""
    download_mnist(path)

    if dataset == "training_data":
        fname_image = os.path.join(path, "train-images.idx3-ubyte").replace("\\", "/")
        fname_label = os.path.join(path, "train-labels.idx1-ubyte").replace("\\", "/")
    elif dataset == "testing_data":
        fname_image = os.path.join(path, "t10k-images.idx3-ubyte").replace("\\", "/")
        fname_label = os.path.join(path, "t10k-labels.idx1-ubyte").replace("\\", "/")
    else:
        raise ValueError("dataset must be 'training_data' or 'testing_data'")

    with open(fname_label, "rb") as flbl:
        _, size = struct.unpack(">II", flbl.read(8))
        lbl = pyarray("b", flbl.read())

    with open(fname_image, "rb") as fimg:
        _, size_img, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = pyarray("B", fimg.read())

    if size != size_img:
        raise ValueError("image count and label count do not match")

    ind = [k for k in range(size) if lbl[k] in digits]
    n = len(ind)
    images = np.zeros((n, rows, cols), dtype=np.float32)
    labels = np.zeros((n, 1), dtype=int)

    for i, idx in enumerate(ind):
        start = idx * rows * cols
        stop = (idx + 1) * rows * cols
        images[i] = np.array(img[start:stop], dtype=np.float32).reshape((rows, cols))
        labels[i] = lbl[idx]

    return images, labels



def data_select(
    train_num: int,
    test_num: int,
    path: str = "./mnist_data",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select only digits 0 and 1, then normalize and one-hot encode."""
    x_train, y_train = load_mnist("training_data", path=path)
    x_test, y_test = load_mnist("testing_data", path=path)

    idx_train = np.append(
        np.where(y_train == 0)[0][:train_num],
        np.where(y_train == 1)[0][:train_num],
    )
    x_train = x_train[idx_train] / 255.0
    y_train = y_train[idx_train]
    y_train = np.eye(2, dtype=np.float32)[y_train].reshape(-1, 2)

    idx_test = np.append(
        np.where(y_test == 0)[0][:test_num],
        np.where(y_test == 1)[0][:test_num],
    )
    x_test = x_test[idx_test] / 255.0
    y_test = y_test[idx_test]
    y_test = np.eye(2, dtype=np.float32)[y_test].reshape(-1, 2)

    return x_train.astype(np.float32), y_train, x_test.astype(np.float32), y_test



def show_samples(x_test: np.ndarray, y_test: np.ndarray, n_samples_show: int = 6) -> None:
    """Display a few 0/1 samples, following the PDF logic."""
    show_count = n_samples_show
    _, axes = plt.subplots(nrows=1, ncols=show_count, figsize=(10, 3))

    for img, targets in zip(x_test, y_test):
        if show_count <= 3:
            break
        if targets[0] == 1:
            axes[show_count - 1].set_title("Labeled: 0")
            axes[show_count - 1].imshow(img.squeeze(), cmap="gray")
            axes[show_count - 1].set_xticks([])
            axes[show_count - 1].set_yticks([])
            show_count -= 1

    for img, targets in zip(x_test, y_test):
        if show_count <= 0:
            break
        if targets[0] == 0:
            axes[show_count - 1].set_title("Labeled: 1")
            axes[show_count - 1].imshow(img.squeeze(), cmap="gray")
            axes[show_count - 1].set_xticks([])
            axes[show_count - 1].set_yticks([])
            show_count -= 1

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2) Single-qubit quantum layer (pyqpanda3 version)
# ---------------------------------------------------------------------------

def circuit(weights: np.ndarray | List[float], shots: int = 100) -> float:
    """
    One-qubit quantum circuit.

    pyqpanda3 official examples construct gates directly with integer qubit/cbit
    indices and run via CPUQVM.run(prog, shots).

    The original PDF code returns the sampled mean of the measurement bit.
    For one qubit, that equals Prob(measurement == '1').
    """
    theta = float(np.array(weights).reshape(-1)[0])

    qc = QCircuit()
    qc << H(0)
    qc << RY(0, theta)

    prog = QProg()
    prog << qc
    prog << measure(0, 0)

    qvm = CPUQVM()
    qvm.run(prog, shots)
    result = qvm.result().get_counts()

    # Same output logic as the original sample: mean of measurement bit.
    counts_1 = result.get("1", 0)
    expectation = counts_1 / float(shots)
    return expectation


# ---------------------------------------------------------------------------
# 3) Hybrid quantum-classical model
# ---------------------------------------------------------------------------
class Hybrid(Module):
    """Quantum layer wrapped as a VQNet Module."""

    def __init__(self, shift: float):
        super(Hybrid, self).__init__()
        self.shift = shift

    def forward(self, input: QTensor) -> QTensor:
        self.input = input
        expectation_z = circuit(np.array(input.data))
        result = [[expectation_z]]
        requires_grad = input.requires_grad

        def _backward(g, input_tensor):
            """Backward pass via symmetric finite-difference / shift-style rule."""
            input_list = np.array(input_tensor.data)
            shift_right = input_list + np.ones(input_list.shape) * self.shift
            shift_left = input_list - np.ones(input_list.shape) * self.shift

            gradients = []
            for i in range(len(input_list)):
                expectation_right = circuit(shift_right[i])
                expectation_left = circuit(shift_left[i])
                gradient = expectation_right - expectation_left
                gradients.append(gradient)

            gradients = np.array([gradients]).T
            return gradients * np.array(g)

        nodes = []
        if input.requires_grad:
            nodes.append(AutoGradNode(tensor=input, df=lambda g: _backward(g, input)))

        return QTensor(data=result, requires_grad=requires_grad, nodes=nodes)


class Net(Module):
    """Hybrid CNN + 1-qubit quantum layer."""

    def __init__(self):
        super(Net, self).__init__() # 继承父类Module的相关属性，super(子类, self).__init__()
        self.conv1 = Conv2D(
            input_channels=1,
            output_channels=6,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding="valid",
        )
        self.maxpool1 = MaxPool2D([2, 2], [2, 2], padding="valid")
        self.conv2 = Conv2D(
            input_channels=6,
            output_channels=16,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding="valid",
        )
        self.maxpool2 = MaxPool2D([2, 2], [2, 2], padding="valid")
        self.fc1 = Linear(input_channels=256, output_channels=64)
        self.fc2 = Linear(input_channels=64, output_channels=1)
        self.hybrid = Hybrid(np.pi / 2)
        self.fc3 = Linear(input_channels=1, output_channels=2)

    def forward(self, x):
        x = F.ReLu()(self.conv1(x))   # -> [B, 6, 24, 24]
        x = self.maxpool1(x)          # -> [B, 6, 12, 12]
        x = F.ReLu()(self.conv2(x))   # -> [B, 16, 8, 8]
        x = self.maxpool2(x)          # -> [B, 16, 4, 4]
        x = tensor.flatten(x, 1)      # -> [B, 256]
        x = F.ReLu()(self.fc1(x))     # -> [B, 64]
        x = self.fc2(x)               # -> [B, 1]
        x = self.hybrid(x)            # -> [B, 1]
        x = self.fc3(x)               # -> [B, 2]
        return x


# ---------------------------------------------------------------------------
# 4) Train / evaluate / visualize
# ---------------------------------------------------------------------------

def train_and_eval(
    train_num: int = 1000,
    test_num: int = 100,
    epochs: int = 10,
    batch_size: int = 1,
    data_path: str = "./mnist_data",
):
    x_train, y_train, x_test, y_test = data_select(train_num, test_num, path=data_path)

    model = Net()   # 训练模式
    optimizer = Adam(model.parameters(), lr=0.005)
    loss_func = CategoricalCrossEntropy()

    train_loss_list: List[float] = []
    val_loss_list: List[float] = []
    train_acc_list: List[float] = []
    val_acc_list: List[float] = []

    for epoch in range(1, epochs):
        total_loss = []
        model.train()
        correct = 0
        n_train = 0

        for x, y in data_generator(x_train, y_train, batch_size=batch_size, shuffle=True):
            x = x.reshape(-1, 1, 28, 28)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_func(y, output)

            loss_np = float(np.array(loss.data))
            np_output = np.array(output.data, copy=False)
            mask = np_output.argmax(1) == y.argmax(1)
            correct += int(np.sum(np.array(mask)))
            n_train += batch_size

            loss.backward()
            optimizer._step()
            total_loss.append(loss_np)

        train_loss_list.append(float(np.sum(total_loss) / len(total_loss)))
        train_acc_list.append(float(np.sum(correct) / n_train))
        print(f"Epoch {epoch:02d} | train loss = {train_loss_list[-1]:.10f} | train acc = {train_acc_list[-1]:.6f}")

        model.eval()    # 切换为评估/验证模式
        correct = 0
        n_eval = 0
        total_loss_eval = []

        for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
            x = x.reshape(-1, 1, 28, 28)
            output = model(x)
            loss = loss_func(y, output)

            loss_np = float(np.array(loss.data))
            np_output = np.array(output.data, copy=False)
            mask = np_output.argmax(1) == y.argmax(1)
            correct += int(np.sum(np.array(mask)))
            n_eval += 1
            total_loss_eval.append(loss_np)

        val_loss_list.append(float(np.sum(total_loss_eval) / len(total_loss_eval)))
        val_acc_list.append(float(np.sum(correct) / n_eval))
        print(f"Epoch {epoch:02d} |  val  loss = {val_loss_list[-1]:.10f} |  val  acc = {val_acc_list[-1]:.6f}")

    return model, (x_train, y_train, x_test, y_test), (
        train_loss_list,
        val_loss_list,
        train_acc_list,
        val_acc_list,
    )



def plot_training_curves(
    train_loss_list: List[float],
    val_loss_list: List[float],
    train_acc_list: List[float],
    val_acc_list: List[float],
    epochs: int,
    ) -> None:
    xrange = range(1, len(train_loss_list) + 1)

    plt.figure()
    figure_path = os.path.join(os.getcwd(), "HQCNN LOSS.png")
    plt.plot(xrange, train_loss_list, color="blue", label="train")
    plt.plot(xrange, val_loss_list, color="red", label="validation")
    plt.title("HQCNN")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(np.arange(1, epochs, step=2))
    plt.legend(loc="upper right")
    plt.savefig(figure_path)
    plt.show()

    plt.figure()
    figure_path = os.path.join(os.getcwd(), "HQCNN Accuracy.png")
    plt.plot(xrange, train_acc_list, color="blue", label="train")
    plt.plot(xrange, val_acc_list, color="red", label="validation")
    plt.title("HQCNN")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(1, epochs, step=2))
    plt.legend(loc="lower right")
    plt.savefig(figure_path)
    plt.show()



def show_predictions(model: Net, x_test: np.ndarray, y_test: np.ndarray, n_samples_show: int = 6) -> None:
    count = 0
    _, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))
    model.eval()

    for x, y in data_generator(x_test, y_test, batch_size=1, shuffle=True):
        if count == n_samples_show:
            break

        x = x.reshape(-1, 1, 28, 28)
        output = model(x)
        pred_label = int(np.argmax(np.array(output.data, copy=False), axis=1)[0])

        axes[count].imshow(x[0].squeeze(), cmap="gray")
        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title(f"Predicted [{pred_label}]")
        count += 1

    plt.tight_layout()
    plt.show()



def main() -> None:
    preview_train_num = 100
    preview_test_num = 50
    train_num = 1000
    test_num = 100
    epochs = 10
    data_path = "./mnist_data"

    # Sample visualization, matching the PDF flow.
    _, _, x_test_preview, y_test_preview = data_select(
        preview_train_num, preview_test_num, path=data_path
    )
    show_samples(x_test_preview, y_test_preview, n_samples_show=6)

    # Train and evaluate.
    model, (_, _, x_test, y_test), stats = train_and_eval(
        train_num=train_num,
        test_num=test_num,
        epochs=epochs,
        batch_size=1,
        data_path=data_path,
    )
    train_loss_list, val_loss_list, train_acc_list, val_acc_list = stats

    # Visualization.
    plot_training_curves(
        train_loss_list,
        val_loss_list,
        train_acc_list,
        val_acc_list,
        epochs=epochs,
    )
    show_predictions(model, x_test, y_test, n_samples_show=6)


if __name__ == "__main__":
    main()
