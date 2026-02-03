r"""
.. _learning_few_data:

Generalization in QML from few training data
============================================

.. meta::
    :property="og:description": Generalization of quantum machine learning models.
    :property="og:image": https://pennylane.ai/qml/_images/few_data_thumbnail.png

.. related::

    tutorial_local_cost_functions Alleviating barren plateaus with local cost functions

*Authors: Korbinian Kottmann, Luis Mantilla Calderon, Maurice Weber — Posted: 29 August 2022*

In this tutorial, we dive into the generalization capabilities of quantum machine learning models.
For the example of a `Quantum Convolutional Neural Network (QCNN) <https://pennylane.ai/qml/glossary/qcnn.html>`_, we show how its generalization error behaves as a
function of the number of training samples. This demo is based on the paper
*"Generalization in quantum machine learning from few training data"*. by Caro et al. [#CaroGeneralization]_.

What is generalization in (Q)ML?
---------------------------------
When optimizing a machine learning model, be it classical or quantum, we aim to maximize its performance over the data
distribution of interest (e.g., images of cats and dogs). However, in practice, we are limited to a finite amount of
data, which is why it is necessary to reason about how our model performs on new, previously unseen data. The difference
between the model's performance on the true data distribution and the performance estimated from our training data is
called the *generalization error*, and it indicates how well the model has learned to generalize to unseen data.
Generalization can be seen as a manifestation of the bias-variance trade-off; models that
perfectly fit the training data admit a low bias at the cost of a higher variance, and hence typically perform poorly on unseen
test data. In the classical machine learning community, this trade-off has been extensively
studied and has led to optimization techniques that favour generalization, for example, by regularizing models via
their variance [#NamkoongVariance]_.
Below, we see a canoncial example of this trade-off, with a model having low bias, but high variance
and therefore high generalization error. The low variance model, on the other hand, has a higher
bias but generalizes better.

.. figure:: /demonstrations/learning_few_data/overfitting.png
    :width: 65%
    :align: center



Let us now dive deeper into generalization properties of quantum machine learning (QML) models. We start by describing
the typical data processing pipeline of a QML model. A classical data input :math:`x` is first encoded in a quantum
state via a mapping :math:`x \mapsto \rho(x)`. This encoded state is then processed through a quantum
channel :math:`\rho(x) \mapsto \mathcal{E}_\alpha(\rho(x))` with learnable parameters :math:`\alpha`. Finally, a measurement is performed on the resulting state
to get the final prediction. Now, the goal is to minimize the expected loss over the data-generating distribution
:math:`P`, indicating how well our model performs on new data. Mathematically, for a loss function :math:`\ell`, the
expected loss, denoted by :math:`R`, is given by

.. math:: R(\alpha) = \mathbb{E}_{(x,y)\sim P}[\ell(\alpha;\,x,\,y)]

where :math:`x` are the features, :math:`y` are the labels, and :math:`P` is their joint distribution.
In practice, as the joint distribution :math:`P` is generally unknown, this quantity has to be
estimated from a finite amount of data. Given a training set :math:`S = \{(x_i,\,y_i)\}_{i=1}^N`
with :math:`N` samples, we estimate the performance of our QML model by calculating the
average loss over the training set

.. math:: R_S(\alpha) = \frac{1}{N}\sum_{i=1}^N \ell(\alpha;\,x_i,\,y_i),

which is referred to as the training loss and is an unbiased estimate of :math:`R(\alpha)`. This is only a proxy
to the true quantity of interest :math:`R(\alpha)`, and their difference is called the generalization error

.. math:: \mathrm{gen}(\alpha) =  R(\alpha) - \hat{R}_S(\alpha),

which is the quantity that we explore in this tutorial. Keeping in mind the bias-variance trade-off, one would expect
that more complex models, i.e. models with a larger number of parameters, achieve a lower error on the training data
but a higher generalization error. Having more training data, on the other hand, leads to a better approximation of the
true expected loss and hence a lower generalization error. This intuition is made precise in Ref. [#CaroGeneralization]_,
where it is shown that :math:`\mathrm{gen}(\alpha)` roughly scales as :math:`\mathcal{O}(\sqrt{T / N})`, where :math:`T`
is the number of parametrized gates and :math:`N` is the number of training samples.
"""

##############################################################################
# Generalization bounds for QML models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# As hinted at earlier, we expect the generalization error to depend both on the richness of the model class, as well as
# on the amount of training data available. As a first result, the authors of Ref. [#CaroGeneralization]_ found that for
# a QML model with at most :math:`T` parametrized local quantum channels, the generalization error depends on :math:`T`
# and :math:`N` according to
#
# .. math:: \mathrm{gen}(\alpha) \sim \mathcal{O}\left(\sqrt{\frac{T\log T}{N}}\right).
#
# We see that this scaling is in line with our intuition that the generalization error scales inversely with the number
# of training samples and increases with the number of parametrized gates. However, as is the case for
# `quantum convolutional neural networks (QCNNs) <https://pennylane.ai/qml/glossary/qcnn.html>`_, it is possible to get a more fine-grained bound by including knowledge on the number of gates :math:`M` which have been reused (i.e. whose parameters are shared across wires). Naively, one could suspect that the generalization error scales as
# :math:`\tilde{\mathcal{O}}(\sqrt{MT/N})` by directly applying the above result (and where
# :math:`\tilde{\mathcal{O}}` includes logarithmic factors). However, the authors of Ref. [#CaroGeneralization]_ found
# that such models actually adhere to the better scaling
#
# .. math:: \mathrm{gen}(\alpha) \sim \mathcal{O}\left(\sqrt{\frac{T\log MT}{N}}\right).
#
# With this, we see that for QCNNs to have a generalization error :math:`\mathrm{gen}(\alpha)\leq\epsilon`, we need a
# training set of size :math:`N \sim T \log MT / \epsilon^2`. For the special case of QCNNs, we can explicitly connect
# the number of samples needed for good generalization to the system size :math:`n` since these models
# use :math:`\mathcal{O}(\log(n))` independently parametrized gates, each of which is used at most :math:`n` times [#CongQuantumCNN]_.
# Putting the pieces together, we find that a training set of size
#
# .. math::  N \sim \mathcal{O}(\mathrm{poly}(\log n))
#
# is sufficient for the generalization error to be bounded by :math:`\mathrm{gen}(\alpha) \leq \epsilon`.
# In the next part of this tutorial, we will illustrate this result by implementing a QCNN to classify different
# digits in the classical ``digits`` dataset. Before that, we set up our QCNN.

##############################################################################
# Quantum convolutional neural networks
# ------------------------------------
# Before we start building a QCNN, let us briefly review the idea of classical CNNs, which have shown
# tremendous success in tasks like image recognition, recommender systems, and sound classification, to name a few.
# For a more in-depth explanation of CNNs, we highly recommend `chapter 9 <https://www.deeplearningbook.org/contents/convnets.html>`_
# in [#DLBook]_.
# Classical CNNs are a family of neural networks which make use of convolutions and pooling operations to
# insert an inductive bias, favouring invariances to spatial transformations like translations, rotations, and scaling.
# A *convolutional layer* consists of a small kernel (a window) that sweeps over a 2D array representation of an image and extracts local
# information while sharing parameters across the spatial dimensions. In addition to the convolutional layers,
# one typically uses pooling layers to reduce the size of the input and to provide a mechanism for summarizing
# information from a neighbourhood of values in the input. On top of reducing dimensionality, these types of layers have the advantage
# of making the model more agnostic to certain transformations like scaling and rotations.
# These two types of layers are applied repeatedly in an alternating manner as shown in the figure below.
#
# .. figure:: /demonstrations/learning_few_data/cnn_pic.png
#     :width: 75%
#     :align: center
#
#     A graphical representation of a CNN. Obtained using Ref. [#LeNailNNSVG]_.
#
# We want to build something similar for a quantum circuit. First, we import the necessary
# libraries we will need in this demo and set a random seed for reproducibility:

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import seaborn as sns
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax  # optimization using jax
import pennylane as qml
import pennylane.numpy as pnp
from scipy.sparse import csr_matrix

sns.set()

seed = 0
rng = np.random.default_rng(seed=seed)


##############################################################################
# To construct a convolutional and pooling layer in a quantum circuit, we will
# follow the QCNN construction proposed in [#CongQuantumCNN]_. The former layer
# will extract local correlations, while the latter allows reducing the dimensionality
# of the feature vector. In a quantum circuit, the convolutional layer, consisting of a kernel swept
# along the entire image, is a two-qubit unitary that correlates neighbouring
# qubits.  As for the pooling layer, we will use a conditioned single-qubit unitary that depends
# on the measurement of a neighboring qubit. Finally, we use a *dense layer* that entangles all
# qubits of the final state using an all-to-all unitary gate as shown in the figure below.
#
# .. figure:: /demonstrations/learning_few_data/qcnn-architecture.png
#     :width: 75%
#     :align: center
#
#     QCNN architecture. Taken from Ref. [#CongQuantumCNN]_.
#
# Breaking down the layers
# --------------------------
#
# The convolutional layer should have the weights of the two-qubit unitary as an input, which are
# updated at every training step.  In PennyLane, we model this arbitrary two-qubit unitary
# with a particular sequence of gates: two single-qubit  :class:`~.pennylane.U3` gates (parametrized by three
# parameters, each), three Ising interactions between both qubits (each interaction is
# parametrized by one parameter), and two additional :class:`~.pennylane.U3` gates on each of the two
# qubits.

ns = 12
num_wires = ns+1
Cn=2
def init_weights():
    """Initializes random weights for the QCNN model."""
    #weights = pnp.random.normal(loc=0, scale=1, size=(21, 3), requires_grad=True)
    weights = pnp.random.uniform(-math.pi/2,math.pi/2,size=(3+Cn*9, 3), requires_grad=True)
    weights_last = pnp.random.uniform(-math.pi/2,math.pi/2,size=15 + num_wires * 3 + (num_wires - 1) * 3, requires_grad=True)
    #weights_last = pnp.random.uniform(-math.pi / 2, math.pi / 2, size=15, requires_grad=True)
    return jnp.array(weights), jnp.array(weights_last)
def first_layer(weights, wires):
    for indx, w in enumerate(wires):
        qml.U3(*weights[indx*3:(indx*3+3)], wires=[w])

    nw = num_wires*3
    n_wires = len(wires)

    for p in [0, 1]:
        for indx, w in enumerate(wires):
            if indx % 2 == p and indx < n_wires - 1:
                qml.IsingXX(weights[nw], wires=[w, wires[indx + 1]])
                qml.IsingYY(weights[nw+1], wires=[w, wires[indx + 1]])
                qml.IsingZZ(weights[nw+2], wires=[w, wires[indx + 1]])
                nw += 3

    #assert nw == len(weights), "this weights is too small!"

def convolutional_layer(weights, wires):
    """Adds a convolutional layer to a circuit.
    Args:
        weights (np.array): 1D array with 15 weights of the parametrized gates.
        wires (list[int]): Wires where the convolutional layer acts on.
        skip_first_layer (bool): Skips the first two U3 gates of a layer.
    """
    n_wires = len(wires)
    assert n_wires >= 3, "this circuit is too small!"

    for p in [0, 1]:
        for indx, w in enumerate(wires):
            if indx % 2 == p and indx < n_wires - 1:
                qml.U3(*weights[:3], wires=[w])
                qml.U3(*weights[3:6], wires=[wires[indx + 1]])
                qml.IsingXX(weights[6], wires=[w, wires[indx + 1]])
                qml.IsingYY(weights[7], wires=[w, wires[indx + 1]])
                qml.IsingZZ(weights[8], wires=[w, wires[indx + 1]])



##############################################################################
# The pooling layer's inputs are the weights of the single-qubit conditional unitaries, which in
# this case are :class:`~.pennylane.U3` gates. Then, we apply these conditional measurements to half of the
# unmeasured wires, reducing our system size by a factor of 2.


def pooling_layer(weights, wires):
    """Adds a pooling layer to a circuit.
    Args:
        weights (np.array): Array with the weights of the conditional U3 gate.
        wires (list[int]): List of wires to apply the pooling layer on.
    """
    n_wires = len(wires)
    assert len(wires) >= 2, "this circuit is too small!"

    for indx, w in enumerate(wires):
        if indx % 2 == 1 and indx < n_wires:
            m_outcome = qml.measure(w)
            qml.cond(m_outcome, qml.U3)(*weights, wires=wires[indx - 1])


##############################################################################
# We can construct a QCNN by combining both layers and using an arbitrary unitary to model
# a dense layer. It will take a set of features — the image — as input, encode these features using
# an embedding map, apply rounds of convolutional and pooling layers, and eventually output the
# desired measurement statistics of the circuit.


def conv_and_pooling(kernel_weights, n_wires):
    """Apply both the convolutional and pooling layer."""
    a=3
    b=3+9
    for n in range(Cn):
        convolutional_layer(kernel_weights[a:b], n_wires)
        a+=9
        b+=9

    pooling_layer(kernel_weights[:3], n_wires)


def dense_layer(weights, wires):
    """Apply an arbitrary unitary gate to a specified set of wires."""
    qml.ArbitraryUnitary(weights, wires)



device = qml.device("default.qubit", wires=num_wires)
nbs = 2 ** ns
index = np.arange(ns + 1)
e2 = 2 ** index
sqrt2 = np.sqrt(2)

@qml.qnode(device, interface="jax")
def conv_net(weights, last_layer_weights, features):
    """Define the QCNN circuit
    Args:
        weights (np.array): Parameters of the convolution and pool layers.
        last_layer_weights (np.array): Parameters of the last dense layer.
        features (np.array): Input data to be embedded using AmplitudEmbedding."""

    layers = weights.shape[1]
    wires = list(range(num_wires))

    # inputs the state input_state
    qml.AmplitudeEmbedding(features=features, wires=wires, pad_with=0.5)
    qml.Barrier(wires=wires, only_visual=True)

    first_layer(last_layer_weights[15:], wires)
    #qml.Barrier(wires=wires, only_visual=True)
    # adds convolutional and pooling layers
    for j in range(layers):
        conv_and_pooling(weights[:, j], wires)
        wires = wires[::2]
        qml.Barrier(wires=wires, only_visual=True)

    assert 15 == 4 ** (len(wires)) - 1, (
        "The size of the last layer weights vector is incorrect!"
        f" \n Expected {4 ** (len(wires)) - 1}, Given {15}"
    )
    dense_layer(last_layer_weights[:15], wires)
    return qml.probs(wires=(0))


# fig, ax = qml.draw_mpl(conv_net)(
#     np.random.rand(3+Gn*9, 3), np.random.rand(15), np.random.rand(2 ** num_wires)
# )
# plt.show()
# plt.show(block=False)

##############################################################################
# In the problem we will address, we need to encode 64 features
# in our quantum state. Thus, we require six qubits (:math:`2^6 = 64`) to encode
# each feature value in the amplitude of each computational basis state.
#
# Training the QCNN on the digits dataset
# ---------------------------------------
# In this demo, we are going to classify the digits ``0`` and ``1`` from the classical ``digits`` dataset.
# Each hand-written digit image is represented as an :math:`8 \times 8` array of pixels as shown below:

#digits = datasets.load_digits()
#images, labels = digits.data, digits.target

#images = images[np.where((labels == 0) | (labels == 1))]
#labels = labels[np.where((labels == 0) | (labels == 1))]

#fig, axes = plt.subplots(nrows=1, ncols=12, figsize=(3, 1))

#for i, ax in enumerate(axes.flatten()):
#    ax.imshow(images[i].reshape((8, 8)), cmap="gray")
#    ax.axis("off")

#plt.tight_layout()
#plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()
#print(images[1, 2], labels[1], type(images), type(labels))


##############################################################################
# For convenience, we create a ``load_digits_data`` function that will make random training and
# testing sets from the ``digits`` dataset from ``sklearn.dataset``:

#Hamiltonian1============================================================================================================
nbs = 2**ns
nbshalf = int(nbs/2)

nz0PXP = 0
ni = 1
for n in range(ns//2):
    nz0PXP += ni
    ni *= 4

ni=1
nz3=0
for n in range(ns//3):
    nz3 += ni
    ni *= 8
if ns//3<ns/3:
    nz3 += ni
print(ns, bin(nz3)[2:].zfill(ns))

DivSqrt2=1/np.sqrt(2)
def eignHInv(H, INvPos):
    # H=HSpa.todense()
    nr = len(H)
    Uinv = np.zeros((nr, nr))
    EInv = np.zeros(nr)
    VInv = np.zeros((nr, nr))
    mBlo1 = 0
    mBlo2 = 0
    for n in range(nr):
        if INvPos[n] == n:
            mBlo2 += 1
    mBlo3=int((mBlo2+nr+1)/2)
    for n in range(nr):
        if INvPos[n] == n:
            Uinv[mBlo1, n] = 1
            mBlo1 += 1
        else:
            nbsbar = INvPos[n]
            if n < nbsbar:
                Uinv[mBlo2, n] = DivSqrt2
                Uinv[mBlo3, n] = DivSqrt2
                Uinv[mBlo2, nbsbar] = DivSqrt2
                Uinv[mBlo3, nbsbar] = -DivSqrt2
                mBlo2 += 1
                mBlo3 += 1
    USpa = csr_matrix(Uinv)
    H = USpa @ H @ USpa.T  # .conj())
    EInv[0:mBlo2], VInv[0:mBlo2, 0:mBlo2] = np.linalg.eigh(H[0:mBlo2, 0:mBlo2])
    EInv[mBlo2:], VInv[mBlo2:, mBlo2:] = np.linalg.eigh(H[mBlo2:, mBlo2:])
    VInv = USpa.T @ VInv
    return EInv, VInv

def vscar1():
    H = np.zeros((nbs, nbs))
    Ueo = np.zeros((nbs, nbs))
    ei = nbshalf  # even index
    oi = 0  # odd index
    ArrInv = np.zeros((nbs, 3), dtype=int)
    # Scar initialize=======================================================================================================
    a = -0.15
    b = -1.8
    a2 = -0.15
    b2 = -1.6
    nz0 = 0
    ni = 1
    vl = np.array([[1, 1], [1, -1]])
    B0 = np.array([[1, 0, 0], [0, 1, 0]])
    B1 = np.array([[0, 0, 0], [sqrt2, 0, sqrt2]])
    C0 = np.array([[0, -1], [1, 0], [0, 0]])
    C1 = np.array([[sqrt2, 0], [0, 0], [-sqrt2, 0]])
    cv = np.zeros((2, 2))
    PsiMPS = np.zeros((nbs, 4), dtype=complex)

    for n in range(ns // 2):
        nz0 += ni
        ni *= 4
    print(ns, bin(nz0)[2:].zfill(ns))

    ni = 1
    nz3 = 0
    for n in range(ns // 3):
        nz3 += ni
        ni *= 8
    if ns // 3 < ns / 3:
        nz3 += ni
    print(ns, bin(nz3)[2:].zfill(ns))

    for n in range(nbs):
        ket = bin(n)[2:].zfill(ns)
        # construct Hamiltonian
        m = 0
        if ket[m + 1] == '0':
            if ket[m] == '0':
                bra = '1' + ket[m + 1:]
            else:
                bra = '0' + ket[m + 1:]

            H[int(bra, 2), n] = 1

        m = ns - 1
        if ket[m - 1] == '0':
            if ket[m] == '0':
                bra = ket[:m] + '1'
            else:
                bra = ket[:m] + '0'

            H[int(bra, 2), n] = 1

        for m in range(1, ns - 1):
            if ket[m - 1] == '0' and ket[m + 1] == '0':
                if ket[m] == '0':
                    bra = ket[:m] + '1' + ket[m + 1:]
                else:
                    bra = ket[:m] + '0' + ket[m + 1:]

                H[int(bra, 2), n] = 1

        # distinguish even and odd==========================================================================
        parity = 0
        for m in range(ns):
            if ket[m] == '1':
                parity += 1

        if parity % 2 == 1:
            Ueo[oi, n] = 1
            ArrInv[n, 0] = oi  # record where it goes
            oi += 1
        else:
            Ueo[ei, n] = 1
            ArrInv[n, 0] = ei  # record where it goes
            ei += 1

        # inverse symmetry====================================================================================
        bra = ket[::-1]
        ArrInv[n, 1] = int(bra, 2)
        ArrInv[n, 2] = n
        # print(bra, ket)

        # calculate the MPS===================================================================================
        id0 = 1
        m = 0
        while m < ns - 1 and id0 == 1:
            if ket[m] == '1' and ket[m + 1] == '1':
                id0 = 0
            m += 1

        if id0 == 1:
            v = np.array([[1, 1], [1, -1]])
            for m in range(ns // 2):
                if ket[m * 2] == '0':
                    v = C0 @ v
                else:
                    v = C1 @ v

                if ket[m * 2 + 1] == '0':
                    v = B0 @ v
                else:
                    v = B1 @ v

            cv = vl @ v
            c2 = cv.reshape(1, 4)
            for m2 in range(4):
                if abs(c2[0, m2]) < 1e-14:
                    c2[0, m2] = 0

            PsiMPS[n, :] = c2

    for n in range(4):
        PsiMPS[:, n] = PsiMPS[:, n] / np.sqrt(sum(PsiMPS[:, n].conj() * PsiMPS[:, n]))
    # end MPS =============================================================================================================

    for n in range(nbs):
        ArrInv[n, 1] = ArrInv[ArrInv[n, 1], 0]

    Uinv = np.zeros((nbshalf, nbshalf))  # /sqrt2
    oi = 0
    nbsqua = int(nbs / 4)
    # sqrt05 = np.sqrt(0.5)
    for n in range(nbs):
        if ArrInv[n, 0] < nbshalf and ArrInv[n, 0] < ArrInv[n, 1]:
            Uinv[oi, ArrInv[n, 0]] = 1
            Uinv[oi, ArrInv[n, 1]] = 1
            Uinv[oi + nbsqua, ArrInv[n, 0]] = 1
            Uinv[oi + nbsqua, ArrInv[n, 1]] = -1
            oi += 1

    HReOd = Ueo @ H @ (Ueo.T.conj())
    H2 = HReOd[:nbshalf, nbshalf:] @ HReOd[nbshalf:, :nbshalf]
    H2inv = Uinv @ H2 @ (Uinv.T.conj())
    H2inv /= 2
    Esy, Vsy = np.linalg.eigh(H2inv[:nbsqua, :nbsqua])
    Easy, Vasy = np.linalg.eigh(H2inv[nbsqua:, nbsqua:])

    Vall = np.zeros((nbs, nbs))
    Eall = np.zeros(nbs)
    Vall[:nbsqua, :nbsqua] = Vsy
    Vall[nbsqua:nbshalf, nbsqua:nbshalf] = Vasy
    Vall[:nbshalf, :nbshalf] = Uinv.T.conj() @ Vall[:nbshalf, :nbshalf]
    Vall[:nbshalf, :nbshalf] /= sqrt2

    Eall[:nbsqua] = Esy
    Eall[nbsqua:nbshalf] = Easy

    for n in range(nbshalf):
        if Eall[n] < 1e-14:
            Eall[n] = 0
            Vall[nbshalf:, n] = Vall[:nbshalf, n].conj()
        else:
            Eall[n] = np.sqrt(Eall[n])
            Vall[nbshalf:, n] = HReOd[nbshalf:, :nbshalf] @ Vall[:nbshalf, n] / Eall[n]

        Vall[:, n] = Vall[:, n] / np.sqrt(sum(Vall[:, n].conj() * Vall[:, n]))
        Vall[nbshalf:, n + nbshalf] = -Vall[nbshalf:, n]

    Eall[nbshalf:] = -Eall[:nbshalf]
    Vall[:nbshalf, nbshalf:] = Vall[:nbshalf, :nbshalf]
    Vall = Ueo.T.conj() @ Vall


    ScarMark = np.zeros(nbs, dtype=int)
    npi= np.zeros(nbs, dtype=int)
    for n in range(nbs):
        if Vall[nz0, n] != 0:
            tem = np.log(Vall[nz0, n].conj() * Vall[nz0, n])
            if tem > a * Eall[n] ** 2 + b:
                ScarMark[n] = 1
            elif tem > -15 and abs(Eall[n]) > 1.415:  # sqrt2
                npi[n] = 1
        #Z3===================================================================================================
        #if Vall[nz3, n] != 0:
        #    tem2 = np.log(Vall[nz3, n].conj() * Vall[nz3, n])
        #    if tem2 > a2 * Eall[n] ** 2 + b2:
        #        ScarMark[n] = 1
        #    elif tem2 > -15 and abs(Eall[n]) > 1.415:  # sqrt2
        #        if ScarMark[n]<0.5:
        #            npi[n] = 1


    nScar = np.sum(ScarMark) + 4
    ntri = np.sum(npi)
    print(nScar,ntri)
    PsiScar = np.zeros((nbs, nScar), dtype=complex)
    EScar = np.zeros(nScar)
    VTi = np.zeros(ntri, dtype=int)
    ni = 4
    ni2=0
    for n in range(nbs):
        if ScarMark[n] == 1:
            PsiScar[:, ni] = Vall[:, n]
            EScar[ni] = Eall[n]
            ni += 1
        if npi[n]==1:
            VTi[ni2] = n
            ni2 += 1

    PsiScar[:, 0:4] = PsiMPS

    EScar[0] = 0
    EScar[1] = sqrt2
    EScar[2] = -sqrt2
    EScar[3] = 0

    overlap=np.real(Vall[nz0PXP, :].conj()*Vall[nz0PXP, :])
    return EScar, PsiScar, nScar, Eall, Vall, VTi,overlap

EScar1, PsiScar1 ,nScar1, Eall1, Vall1,VTi,Overlap = vscar1()

EScar2, PsiScar2 ,nScar2, Eall2, Vall2,VTi,Overlap = vscar1()


#return EScar, PsiScar, nScar, Eall, Vall, VTi, OverlapKxi,ScarMarkerZXZ,ndwArray


#EScar1, PsiScar1, nScar1, Eall1, Vall1, VTi, OverlapKxi,ScarMarkerZXZ,ndw = vscar1()
#
# nScar2=nScar1
# PsiScar2=PsiScar1

#========================================================================================================================
AnciBit=[1,0]
def load_digits_data(num_train, num_test, rng):
    """Return training and testing data of digits dataset."""
    q0 = np.zeros((nbs, nbs), dtype=complex)
    q1 = np.zeros((nbs, nbs), dtype=complex)
    x_train = np.zeros([num_train, nbs*2], dtype=complex)
    y_train = np.zeros(num_train, dtype=int)
    #choose trivial eigenstates
    lenVTi=len(VTi)
    nV=int(num_train* 0.5 / 2)
    #nV = int(num_train/ 2)
    if lenVTi>nV :
        ntrain1=nV
    else:
        ntrain1=lenVTi

    RamVTi=np.random.choice(VTi,ntrain1,replace=False)
    for ml in range(ntrain1):
        # ra = np.random.rand(nbs, 2)
        # for nl in range(nbs):
        #     psi[nl] = ra[nl, 0] * np.exp(2j * math.pi * ra[nl, 1])
        # psi/=(nbs**2)
        psi=Vall1[:,RamVTi[ml]]
        # for nl in range(nScar1):
        #     psi -= PsiScar1[:, nl] * sum(PsiScar1[:, nl].conj() * psi)
        # psi /= np.sqrt(sum(psi.conj() * psi))
        x_train[ml] = np.kron(psi,AnciBit)
        y_train[ml] = 0

    psi = np.zeros(nbs, dtype=complex)
    #choose thermal states
    for ml in range(ntrain1,num_train // 2):
        ra = np.random.rand(nbs, 2)
        for nl in range(nbs):
            psi[nl] = ra[nl, 0] * np.exp(2j * math.pi * ra[nl, 1])

        for nl in range(nScar1):
            psi -= PsiScar1[:, nl] * sum(PsiScar1[:, nl].conj() * psi)

        psi /= np.sqrt(sum(psi.conj()*psi))
        x_train[ml] = np.kron(psi,AnciBit)
        y_train[ml] = 0
        q = psi.reshape(nbs, 1) @ np.conj(psi).reshape(1, nbs)
        q0 = q0+q
    #prepar the scar states====================================================================
    for ml in range(num_train // 2, num_train):
        nTemScar = ml-num_train // 2
        if nTemScar < nScar1:
            ra2 = np.zeros([nScar1, 2])
            ra2[nTemScar, 0] = 1
        else:
            ra2 = np.random.rand(nScar1, 2)

        psi = np.zeros(nbs, dtype=complex)
        for nl in range(0, nScar1):
            psi += PsiScar1[:, nl]*ra2[nl, 0] * np.exp(2j * math.pi * ra2[nl, 1])

        psi /= np.sqrt(sum(psi.conj() * psi))
        x_train[ml] = np.kron(psi,AnciBit)
        y_train[ml] = 1
        q = psi.reshape(nbs, 1)@np.conj(psi).reshape(1, nbs)
        q1 = q1+q

    # q0 = q0/(num_train//2)
    # q1 = q1/(num_train-num_train//2)
    # q = q0*(num_train-num_train//2)/num_train-q1*(num_train//2)/num_train
    # q = q.T.conj()@q
    # d = np.linalg.eigvalsh(q)
    # for nl in range(nbs):
    #     if d[nl] < 0:
    #         d[nl] = 0
    #
    # d = np.sqrt(d)
    # print(num_train, 0.5-0.5*np.sum(d))

    x_test = np.zeros([num_test, nbs*2], dtype=complex)
    y_test = np.zeros(num_test, dtype=int)
    for ml in range(num_test // 2):
        ra = np.random.rand(nbs, 2)
        for nl in range(nbs):
            psi[nl] = ra[nl, 0] * np.exp(2j * math.pi * ra[nl, 1])

        for nl in range(nScar2):
            psi -= PsiScar2[:, nl] * sum(PsiScar2[:, nl].conj() * psi)

        psi /= np.sqrt(sum(psi.conj()*psi))
        x_test[ml] = np.kron(psi,AnciBit)
        y_test[ml] = 0

    for ml in range(num_test // 2, num_test):
        # nTemScar = ml-num_test // 2
        # if nTemScar < nScar2:
        #     ra2 = np.zeros([nScar2, 2])
        #     ra2[nTemScar, 0] = 1
        # else:
        #     ra2 = np.random.rand(nScar2, 2)
        ra2 = np.random.rand(nScar2, 2)
        psi = np.zeros(nbs, dtype=complex)
        for nl in range(0, nScar2):
            psi += PsiScar2[:, nl]*ra2[nl, 0] * np.exp(2j * math.pi * ra2[nl, 1])

        psi /= np.sqrt(sum(psi.conj() * psi))
        x_test[ml] = np.kron(psi,AnciBit)
        y_test[ml] = 1

    return (
        jnp.asarray(x_train),
        jnp.asarray(y_train),
        jnp.asarray(x_test),
        jnp.asarray(y_test),
    )


##############################################################################
# To optimize the weights of our variational model, we define the cost and accuracy functions
# to train and quantify the performance on the classification task of the previously described QCNN:


@jax.jit
def compute_out(weights, weights_last, features, labels):
    """Computes the output of the corresponding label in the qcnn"""
    cost = lambda weights, weights_last, feature, label: conv_net(weights, weights_last, feature)[
        label
    ]
    return jax.vmap(cost, in_axes=(None, None, 0, 0), out_axes=0)(
        weights, weights_last, features, labels
    )


def compute_accuracy(weights, weights_last, features, labels):
    """Computes the accuracy over the provided features and labels"""
    out = compute_out(weights, weights_last, features, labels)
    return jnp.sum(out > 0.5) / len(out)


def compute_cost(weights, weights_last, features, labels):
    """Computes the cost over the provided features and labels"""
    out = compute_out(weights, weights_last, features, labels)
    return 1.0 - jnp.sum(out) / len(labels)

value_and_grad = jax.jit(jax.value_and_grad(compute_cost, argnums=[0, 1]))


##############################################################################
# We are going to perform the classification for training sets with different values of :math:`N`. Therefore, we
# define the classification procedure once and then perform it for different datasets.
# Finally, we update the weights using the :class:`pennylane.AdamOptimizer` and use these updated weights to
# calculate the cost and accuracy on the testing and training set:


def train_qcnn(n_train, n_test, n_epochs):
    """
    Args:
        n_train  (int): number of training examples
        n_test   (int): number of test examples
        n_epochs (int): number of training epochs
        desc  (string): displayed string during optimization

    Returns:
        dict: n_train,
        steps,
        train_cost_epochs,
        train_acc_epochs,
        test_cost_epochs,
        test_acc_epochs

    """
    # load data
    x_train, y_train, x_test, y_test = load_digits_data(n_train, n_test, rng)

    # init weights and optimizer
    weights, weights_last = init_weights()

    # learning rate decay
    cosine_decay_scheduler = optax.cosine_decay_schedule(0.05, decay_steps=n_epochs, alpha=0.9)
    optimizer = optax.adam(learning_rate=cosine_decay_scheduler)
    opt_state = optimizer.init((weights, weights_last))

    # data containers
    train_cost_epochs, test_cost_epochs, train_acc_epochs, test_acc_epochs = [], [], [], []

    for step in range(n_epochs):
        # Training step with (adam) optimizer
        train_cost, grad_circuit = value_and_grad(weights, weights_last, x_train, y_train)
        updates, opt_state = optimizer.update(grad_circuit, opt_state)
        weights, weights_last = optax.apply_updates((weights, weights_last), updates)

        train_cost_epochs.append(train_cost)

        # compute accuracy on training data
        train_acc = compute_accuracy(weights, weights_last, x_train, y_train)
        train_acc_epochs.append(train_acc)

        # compute accuracy and cost on testing data
        test_out = compute_out(weights, weights_last, x_test, y_test)
        test_acc = jnp.sum(test_out > 0.5) / len(test_out)
        test_acc_epochs.append(test_acc)
        test_cost = 1.0 - jnp.sum(test_out) / len(test_out)
        test_cost_epochs.append(test_cost)

    # return weights, weights_last, dict(
    #     n_train=[n_train] * n_epochs,
    #     step=np.arange(1, n_epochs + 1, dtype=int),
    #     train_cost=train_cost_epochs,
    #     train_acc=train_acc_epochs,
    #     test_cost=test_cost_epochs,
    #     test_acc=test_acc_epochs
    # )
    return dict(
        n_train=[n_train] * n_epochs,
        step=np.arange(1, n_epochs + 1, dtype=int),
        train_cost=train_cost_epochs,
        train_acc=train_acc_epochs,
        test_cost=test_cost_epochs,
        test_acc=test_acc_epochs,
        weights=weights,
        weights_last=weights_last,
    )



##############################################################################
# .. note::
#
#     There are some small intricacies for speeding up this code that are worth mentioning. We are using ``jax`` for our training
#     because it allows for `just-in-time <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_ (``jit``) compilation. A function decorated with ``@jax.jit`` will be compiled upon its first execution
#     and cached for future executions. This means the first execution will take longer, but all subsequent executions are substantially faster.
#     Further, we use ``jax.vmap`` to vectorize the execution of the QCNN over all input states, as opposed to looping through the training and test set at every execution.

##############################################################################
# Training for different training set sizes yields different accuracies, as seen below. As we increase the training data size, the overall test accuracy,
# a proxy for the models' generalization capabilities, increases:

n_test = 50
#n_epochs = 100
n_epochs = 400#300
#n_reps = 100
n_reps = 1#30#200 #<---------------------------------------------------------------------------------------------------------------------------------------------

#train_sizes = [3]
train_sizes = [200] #200

def run_iterations(n_train):
    # results_df = pd.DataFrame(
    #     columns=["train_acc", "train_cost", "test_acc", "test_cost", "step", "n_train"]
    # )
    LossBar = 100.0

    for nr in range(n_reps):
        results = train_qcnn(n_train=n_train, n_test=n_test, n_epochs=n_epochs)
        TraCost = results["train_cost"]
        if TraCost[-1] < LossBar:
            LossBar = TraCost[-1]
            SmaRes = results
            print(SmaRes["n_train"][0], nr, LossBar)

        # results_df = pd.concat(
        #     [results_df, pd.DataFrame.from_dict(results)], axis=0, ignore_index=True
        # )

    return SmaRes


# run training for multiple sizes=====================================================================================


colors = sns.color_palette()
results_df = run_iterations(n_train=train_sizes[0])
SmaRes= run_iterations(n_train=train_sizes[0])
figmin, axsmin = plt.subplots(2, 3, figsize=(16.5, 10))
c1=int(0)
axsmin[0, 0].plot(SmaRes["step"], SmaRes["train_cost"], "o-", label=fr"$N={train_sizes[0]}$", markevery=10, color=colors[c1],
         alpha=0.8)
axsmin[0, 0].plot(SmaRes["step"], SmaRes["test_cost"], "x--", label=None, markevery=10, color=colors[c1], alpha=0.8)
axsmin[0, 1].plot(SmaRes["step"], SmaRes["train_acc"], "o-", label=fr"$N={train_sizes[0]}$", markevery=10,
                  color=colors[c1],
                  alpha=0.8)
axsmin[0, 1].plot(SmaRes["step"], SmaRes["test_acc"], "x--", label=None, markevery=10, color=colors[c1], alpha=0.8)
for n_train in train_sizes[1:]:
    SmaWe= run_iterations(n_train=n_train)
    c1+=1
    axsmin[0, 0].plot(SmaRes["step"], SmaRes["train_cost"], "o-", label=fr"$N={n_train}$", markevery=10, color=colors[c1], alpha=0.8)
    axsmin[0, 0].plot(SmaRes["step"], SmaRes["test_cost"], "x--", label=None, markevery=10, color=colors[c1], alpha=0.8)
    axsmin[0, 1].plot(SmaRes["step"], SmaRes["train_acc"], "o-", label=fr"$N={n_train}$", markevery=10, color=colors[c1],
            alpha=0.8)
    axsmin[0, 1].plot(SmaRes["step"], SmaRes["test_acc"], "x--", label=None, markevery=10, color=colors[c1], alpha=0.8)
    plt.savefig('name')

legend_elements = [
    mpl.lines.Line2D([0], [0], label=f'N={n}', color=colors[i]) for i, n in enumerate(train_sizes)
    ] + [
    mpl.lines.Line2D([0], [0], marker='o', ls='-', label='Train', color='Black'),
    mpl.lines.Line2D([0], [0], marker='x', ls='--', label='Test', color='Black')
    ]

axsmin[0, 0].legend(handles=legend_elements, ncol=3)
#Classify the eignstates=====================================================================================
ScarMarker2=np.zeros(nbs,dtype=int)
ScarMarker3=np.zeros(nbs,dtype=int)
for n in range(nbs):
    PCS=conv_net(SmaRes["weights"], SmaRes["weights_last"], np.kron(Vall1[:,n],[1, 0]))[0]
    if PCS >0.5:
        linesScar = "o"
        print(PCS)
    else:
        linesScar = "x"
        ScarMarker2[n]=1

    if Overlap[n] != 0:
        logq = np.log(Overlap[n])
        axsmin[1, 0].plot(Eall1[n], logq, linesScar)
        if logq>-20 and ScarMarker2[n]==1:
            ScarMarker3[n] = 1


axsmin[1, 0].set_ylim(-20, 0)

# np.savez('20231209',SmaWe=SmaWe,SmaWeL=SmaWeL,ns=ns,ScarMarker2=ScarMarker2,ScarMarker3=ScarMarker3,SmaRes=SmaRes,OverlapKxi=OverlapKxi, JLam = 1,JSig = 0.1,Jzz = 1)
# data2=np.load('20241008GC2PXPOBC_12bit.npz')
# if data2["train_cost"][-1]>SmaRes["train_cost"][-1]:
np.savez('20241008GC2PXPOBC_12bit',
         ns=ns,ScarMarker2=ScarMarker2,ScarMarker3=ScarMarker3,Overlap=Overlap,n_reps=n_reps,
        n_epochs=n_epochs,train_sizes = train_sizes,
         n_train=SmaRes["n_train"],
         step=SmaRes["step"],
         train_cost=SmaRes["train_cost"],
         train_acc=SmaRes["train_acc"],
         test_cost=SmaRes["test_cost"],
         test_acc=SmaRes["test_acc"],
         weights=SmaRes["weights"],
         weights_last=SmaRes["weights_last"]
         )


nS1=sum(ScarMarker2)
nS2=sum(ScarMarker3)
ES1=np.zeros(nS1)
ES2=np.zeros(nS2)


ni1=0
ni2=0
for n in range(nbs):
    if ScarMarker2[n]==1:
        ES1[ni1]=Eall1[n]
        ni1+=1
    if ScarMarker3[n]==1:
        ES2[ni2]=Eall1[n]
        ni2 += 1

ES1=sorted(ES1)
ES2=sorted(ES2)

El=np.arange(-7,7,0.01)
dEl1=np.zeros(len(El))
dEl2=np.zeros(len(El))
wid=0.1
w1=np.sqrt(2*math.pi)*wid
w2=2*wid**2
for n in range(len(El)):
    for m in range(nS1):
        dEl1[n]+= math.exp(-(ES1[m]-El[n])**2/w2)/w1
    for m in range(nS2):
        dEl2[n]+= math.exp(-(ES2[m]-El[n])**2/w2)/w1

axsmin[1, 1].plot(El,dEl1)
axsmin[1, 2].plot(El,dEl2)

plt.show()

