"""
1D-Inception-ResNet for NIR quantitative analysis (regression).
Implementation by: D.Passos (dmpassos@uagl.pt)

Paper:
A. Tan et al., "1D-inception-resnet for NIR quantitative analysis and its transferability
between different spectrometers", Infrared Physics & Technology 129 (2023) 104559.

Implementation notes (from paper text + Table 2):
- Convolution blocks use BatchNorm + ELU activation.
- Inception-Residual module uses 4 parallel conv paths whose outputs are concatenated (Concat),
  then merged with a residual/identity path (Add).
- Fully-connected layers use Softplus activation.
- Output is Dense(1) with linear activation (regression).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2


def inception_residual_module_1d(
    x: tf.Tensor,
    *,
    # C3, C4, C5->C6, C7->C8 (paper: features spliced from Conv3, Conv4, Conv6, Conv8)
    c3_filters: int,
    c3_kernel: int,
    c4_filters: int,
    c4_kernel: int,
    c5_filters: int,
    c5_kernel: int,
    c6_filters: int,
    c6_kernel: int,
    c7_filters: int,
    c7_kernel: int,
    c8_filters: int,
    c8_kernel: int,
    stride: int = 1,
    l2_regularization: float = 1e-4,
    name: str = "InceptionResidual",
) -> tf.Tensor:
    """Inception-Residual module: Concat(branches) + residual Add(identity/projection)."""

    reg = l2(l2_regularization)

    def conv_bn_elu(t: tf.Tensor, filters: int, kernel_size: int, strides: int, lname: str) -> tf.Tensor:
        t = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            kernel_regularizer=reg,
            kernel_initializer="he_normal",
            name=lname,
        )(t)
        t = layers.BatchNormalization(name=f"{lname}_bn")(t)
        t = layers.Activation("elu", name=f"{lname}_elu")(t)
        return t

    with tf.name_scope(name):
        # Branch 1: C3
        b1 = conv_bn_elu(x, c3_filters, c3_kernel, stride, f"{name}_C3")

        # Branch 2: C4
        b2 = conv_bn_elu(x, c4_filters, c4_kernel, stride, f"{name}_C4")

        # Branch 3: C5 (1x1) -> C6 (k)
        b3 = conv_bn_elu(x, c5_filters, c5_kernel, 1, f"{name}_C5")
        b3 = conv_bn_elu(b3, c6_filters, c6_kernel, stride, f"{name}_C6")

        # Branch 4: C7 (1x1) -> C8 (k)
        b4 = conv_bn_elu(x, c7_filters, c7_kernel, 1, f"{name}_C7")
        b4 = conv_bn_elu(b4, c8_filters, c8_kernel, stride, f"{name}_C8")

        concat = layers.Concatenate(axis=-1, name=f"{name}_concat")([b1, b2, b3, b4])

        # Residual add: if channel mismatch, project with 1x1 conv
        shortcut = x
        if shortcut.shape[-1] != concat.shape[-1]:
            shortcut = layers.Conv1D(
                filters=int(concat.shape[-1]),
                kernel_size=1,
                strides=1,
                padding="same",
                kernel_regularizer=reg,
                kernel_initializer="he_normal",
                name=f"{name}_proj",
            )(shortcut)
            shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

        out = layers.Add(name=f"{name}_add")([concat, shortcut])
        out = layers.Activation("elu", name=f"{name}_out_elu")(out)
        return out


def inception_resnet_1d_model(
    input_vector_dimension: int,
    # C1..C8 kernels
    c1_kernel: int = 1,
    c2_kernel: int = 3,
    c3_kernel: int = 1,
    c4_kernel: int = 1,
    c5_kernel: int = 1,
    c6_kernel: int = 3,
    c7_kernel: int = 1,
    c8_kernel: int = 5,
    # C1..C8 filters (paper calls "Hidden number (C1 to C8)")
    c1_filters: int = 96,
    c2_filters: int = 256,
    c3_filters: int = 64,
    c4_filters: int = 64,
    c5_filters: int = 64,
    c6_filters: int = 64,
    c7_filters: int = 64,
    c8_filters: int = 64,
    # F1..F3 (F4 is output=1)
    f1_units: int = 256,
    f2_units: int = 256,
    f3_units: int = 256,
    # regularization / dropout
    dropout_rate: float = 0.15,
    l2_regularization: float = 1e-4,
    stride: int = 1,
    name: str = "1DInceptionResNet",
) -> tf.keras.Model:
    """
    Build 1D-Inception-ResNet (regression).

    Input: (input_vector_dimension, 1)
    Output: Dense(1, linear)
    """

    reg = l2(l2_regularization)

    inputs = layers.Input(shape=(input_vector_dimension, 1), name="input")

    # Paper mentions a batch standardization layer early in the network
    x = layers.BatchNormalization(name="input_bn")(inputs)

    # C1
    x = layers.Conv1D(
        filters=c1_filters,
        kernel_size=c1_kernel,
        strides=stride,
        padding="same",
        kernel_regularizer=reg,
        kernel_initializer="he_normal",
        name="C1",
    )(x)
    x = layers.BatchNormalization(name="C1_bn")(x)
    x = layers.Activation("elu", name="C1_elu")(x)

    # C2
    x = layers.Conv1D(
        filters=c2_filters,
        kernel_size=c2_kernel,
        strides=stride,
        padding="same",
        kernel_regularizer=reg,
        kernel_initializer="he_normal",
        name="C2",
    )(x)
    x = layers.BatchNormalization(name="C2_bn")(x)
    x = layers.Activation("elu", name="C2_elu")(x)

    # Inception-Residual module producing (Concat + Add) structure
    x = inception_residual_module_1d(
        x,
        c3_filters=c3_filters, c3_kernel=c3_kernel,
        c4_filters=c4_filters, c4_kernel=c4_kernel,
        c5_filters=c5_filters, c5_kernel=c5_kernel,
        c6_filters=c6_filters, c6_kernel=c6_kernel,
        c7_filters=c7_filters, c7_kernel=c7_kernel,
        c8_filters=c8_filters, c8_kernel=c8_kernel,
        stride=stride,
        l2_regularization=l2_regularization,
        name="InceptionResidual",
    )

    # Paper: dropout after the Inception-Residual module (15%)
    x = layers.Dropout(dropout_rate, name="inception_dropout")(x)

    # Flatten + Fully connected layers (Softplus)
    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(f1_units, kernel_regularizer=reg, kernel_initializer="he_normal", name="F1")(x)
    x = layers.Activation("softplus", name="F1_softplus")(x)

    x = layers.Dense(f2_units, kernel_regularizer=reg, kernel_initializer="he_normal", name="F2")(x)
    x = layers.Activation("softplus", name="F2_softplus")(x)

    x = layers.Dense(f3_units, kernel_regularizer=reg, kernel_initializer="he_normal", name="F3")(x)
    x = layers.Activation("softplus", name="F3_softplus")(x)

    # Output (regression): ALWAYS linear activation as requested
    outputs = layers.Dense(1, activation="linear", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name=name)
    return model


# =============================================================================
# Paper hyperparameters / training setup (from Table 2 + Section 2.2 from the paper): 
# =============================================================================
#
# Two task-specific configurations were used in the paper (Table 2):
#
# ADF task:
#   Kernel sizes (C1..C8):  [1, 3, 1, 1, 1, 3, 1, 5]
#   Filters     (C1..C8):  [96, 256, 64, 64, 64, 64, 64, 64]
#   Fully-connected (F1..F4): [256, 256, 256, 1]
#
# IVOMD task:
#   Kernel sizes (C1..C8):  [1, 3, 1, 1, 1, 5, 1, 7]
#   Filters     (C1..C8):  [64, 256, 64, 64, 32, 64, 32, 64]
#   Fully-connected (F1..F4): [256, 256, 256, 1]
#
# Shared (both tasks):
#   Stride: 1
#   Dropout: 0.15 (applied after the Inception-Residual module output)
#   Regularization coefficient (L2): 1e-4
#   Conv blocks: BatchNorm after Conv1D; activation = ELU
#   FC activation: Softplus
#   Loss: RMSE + L2 regularization term (paper formula (2)); optimizer Adam
#   Early stopping: stop if validation loss >= min loss for 50 consecutive epochs
#   Learning-rate decay reported: 0.1
#
# Note:
# - This builder returns an uncompiled model (consistent with "model factory" usage).
# - To mirror the paper’s loss literally, you can compile with a RMSE loss and rely on
#   kernel_regularizer=l2(...) to contribute the L2 penalty to the total loss.
#   Example:
#       model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=...),
#                    loss=tf.keras.metrics.RootMeanSquaredError())
#
