"""
ResidualSpectra model factory — TensorFlow/Keras >= 2.20 (regression adaptation)
Implementation by: D.Passos (dmpassos@uagl.pt)

Paper:
Xin Wang et al., "End-to-end analysis modeling of vibrational spectroscopy based on deep learning approach",
Journal of Chemometrics (2020) e3291.

Important:
- The original paper proposes ResidualSpectra for *classification* (Dense + Softmax, cross-entropy loss).
- For the benchmark, we adapt it to *regression* by using: Dense(1, activation="linear") as the output layer.

Architecture (Figure 1 + Section 2.2):
Input -> Conv1 -> Inception(Add1) -> Residual block(Add2) -> Flatten -> Dropout -> Dense(out)

Implementation choices:
- Input is (L, 1) (1D spectrum with channel=1).
- Default padding='same' to match the paper's use of SAME convolution mode mentioned for feature-map visualization.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    ReLU,
)


__all__ = ["residualspectra_model"]


def _conv1d(
    x: tf.Tensor,
    *,
    filters: int,
    kernel_size: int,
    strides: int,
    padding: str,
    activation: str,
    name: str,
) -> tf.Tensor:
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer="he_normal",
        name=name,
    )(x)
    if activation.lower() == "leakyrelu":
        x = LeakyReLU(name=f"{name}_lrelu")(x)
    elif activation.lower() == "relu":
        x = ReLU(name=f"{name}_relu")(x)
    elif activation.lower() == "linear" or activation is None:
        pass
    else:
        x = tf.keras.layers.Activation(activation, name=f"{name}_{activation}")(x)
    return x


def residualspectra_model(
    input_vector_dimension: int,
    *,
    input_channels: int = 1,
    kernel_size: int = 8,
    padding: str = "same",
    # Conv1
    conv1_filters: int = 64,
    conv1_stride: int = 1,
    conv1_activation: str = "leakyrelu",
    # Inception block
    conv2_filters: int = 32,
    conv2_stride: int = 2,
    conv2_activation: str = "relu",
    conv3_filters: int = 32,
    conv3_stride: int = 1,
    conv3_activation: str = "leakyrelu",
    conv4_filters: int = 32,
    conv4_stride: int = 2,
    conv4_activation: str = "relu",
    # Residual block
    conv5_filters: int = 32,
    conv5_stride: int = 1,
    conv5_activation: str = "relu",
    conv6_filters: int = 32,
    conv6_stride: int = 1,
    conv6_activation: str = "relu",
    residual_post_add_activation: str = "relu",
    # Head
    dropout_rate: float = 0.5,
    name: str = "ResidualSpectra",
) -> Model:
    """
    Build ResidualSpectra adapted for regression.

    Parameters
    ----------
    input_vector_dimension : int
        Number of spectral variables (length L).
    input_channels : int
        Usually 1 (spectra shaped (L,1)).
    kernel_size : int
        Convolution kernel size. The paper figure indicates kernel size 8 (Conv(1,8)).
    padding : str
        Default 'same' (paper mentions SAME convolution mode for stride=1 layers).
    dropout_rate : float
        Dropout after Flatten. Paper uses 0.5.
    """
    if input_vector_dimension <= 0:
        raise ValueError("input_vector_dimension must be a positive integer.")

    inputs = Input(shape=(input_vector_dimension, input_channels), name="input")

    # --- Conv1 ---
    x = _conv1d(
        inputs,
        filters=conv1_filters,
        kernel_size=kernel_size,
        strides=conv1_stride,
        padding=padding,
        activation=conv1_activation,
        name="Conv1",
    )

    # --- Inception block (two paths merged by Add1) ---
    # Path A: Conv3 (stride 1) -> Conv4 (stride 2)
    pA = _conv1d(
        x,
        filters=conv3_filters,
        kernel_size=kernel_size,
        strides=conv3_stride,
        padding=padding,
        activation=conv3_activation,
        name="Conv3",
    )
    pA = _conv1d(
        pA,
        filters=conv4_filters,
        kernel_size=kernel_size,
        strides=conv4_stride,
        padding=padding,
        activation=conv4_activation,
        name="Conv4",
    )

    # Path B: Conv2 (stride 2)
    pB = _conv1d(
        x,
        filters=conv2_filters,
        kernel_size=kernel_size,
        strides=conv2_stride,
        padding=padding,
        activation=conv2_activation,
        name="Conv2",
    )

    add1 = Add(name="Add1")([pA, pB])

    # --- Residual block (Conv5 -> Conv6 -> Add2 with shortcut add1) ---
    r = _conv1d(
        add1,
        filters=conv5_filters,
        kernel_size=kernel_size,
        strides=conv5_stride,
        padding=padding,
        activation=conv5_activation,
        name="Conv5",
    )
    r = _conv1d(
        r,
        filters=conv6_filters,
        kernel_size=kernel_size,
        strides=conv6_stride,
        padding=padding,
        activation=conv6_activation,
        name="Conv6",
    )

    add2 = Add(name="Add2")([r, add1])

    if residual_post_add_activation is not None:
        if residual_post_add_activation.lower() == "relu":
            add2 = ReLU(name="Add2_relu")(add2)
        elif residual_post_add_activation.lower() == "leakyrelu":
            add2 = LeakyReLU(name="Add2_lrelu")(add2)
        elif residual_post_add_activation.lower() == "linear":
            pass
        else:
            add2 = tf.keras.layers.Activation(residual_post_add_activation, name="Add2_act")(add2)

    # --- Head ---
    x = Flatten(name="Flatten")(add2)
    x = Dropout(rate=dropout_rate, name="Dropout")(x)

    # Regression adaptation (per benchmark guideline)
    outputs = Dense(1, activation="linear", name="Output")(x)

    return Model(inputs=inputs, outputs=outputs, name=name)


# =============================================================================
# Paper hyperparameters / training setup (classification in the paper)
# =============================================================================
#
# Architecture parameters (Section 2.2 + Figure 1):
# - Conv1: 64 filters, stride 1; activation shown as LeakyReLU.
# - Inception:
#     * Conv2: 32 filters, stride 2; activation ReLU
#     * Conv3: 32 filters, stride 1; activation LeakyReLU
#     * Conv4: 32 filters, stride 2; activation ReLU
#     * Add1: element-wise add of the two paths.
# - Residual block:
#     * Conv5: 32 filters, stride 1; activation ReLU
#     * Conv6: 32 filters, stride 1; activation ReLU
#     * Add2: residual add with identity shortcut; ReLU after addition (Figure 2).
# - Flatten -> Dropout(0.5) -> Dense -> Softmax (classification).
# - Kernel size: the figure labels Conv(1,8,...) which indicates kernel size 8.
#
# Training (Section 2.2):
# - Loss: cross-entropy (classification).
# - Optimizer: Adam with learning rate 0.001, beta1=0.9, beta2=0.999, epsilon=1e-8.
#
# Benchmark adaptation here:
# - Output is Dense(1, linear) for regression.
# =============================================================================
