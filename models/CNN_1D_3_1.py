"""
1D-CNN_3_1 baseline model factory for NIRBENCH.

Stacked 1D CNN with three convolutional blocks and one dense regression layer.
"""

from __future__ import annotations

from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    MaxPooling1D,
)


__all__ = ["build_cnn_1d_3_1"]


def build_cnn_1d_3_1(
    input_vector_dimension: int,
    *,
    input_channels: int = 1,
    dropout_rate: float = 0.20,
    l2_coeff: float = 1e-3,
    name: str = "1D-CNN_3_1",
) -> Model:
    """
    Build the 1D-CNN_3_1 regression baseline.

    Architecture:
      Input -> Conv1D(16, 21) -> MaxPool ->
      Conv1D(32, 11) -> MaxPool ->
      Conv1D(64, 5) -> GAP -> Dropout ->
      Dense(64) -> Dense(1)
    """
    if input_vector_dimension <= 0:
        raise ValueError("input_vector_dimension must be a positive integer.")
    if input_channels <= 0:
        raise ValueError("input_channels must be a positive integer.")

    reg = regularizers.l2(float(l2_coeff))

    inputs = Input(shape=(input_vector_dimension, input_channels), name="input")
    x = Conv1D(
        filters=16,
        kernel_size=21,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=reg,
        name="conv1",
    )(inputs)
    x = MaxPooling1D(pool_size=2, name="pool1")(x)
    x = Conv1D(
        filters=32,
        kernel_size=11,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=reg,
        name="conv2",
    )(x)
    x = MaxPooling1D(pool_size=2, name="pool2")(x)
    x = Conv1D(
        filters=64,
        kernel_size=5,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=reg,
        name="conv3",
    )(x)
    x = GlobalAveragePooling1D(name="gap")(x)
    x = Dropout(rate=dropout_rate, name="dropout")(x)
    x = Dense(
        64,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=reg,
        name="dense1",
    )(x)
    outputs = Dense(
        1,
        activation="linear",
        kernel_initializer="he_normal",
        kernel_regularizer=reg,
        name="output",
    )(x)

    return Model(inputs=inputs, outputs=outputs, name=name)


# Rationale:
# - This is the standard sequential multi-block 1D CNN baseline: deeper than
#   the single-convolution models, but still a conventional vanilla CNN.
#
# References:
# - Malek et al. (2017), "One-dimensional convolutional neural networks for
#   spectroscopic signal regression"
# - Kawamura (2021), "Using a One-Dimensional Convolutional Neural Network on
#   Visible and Near-Infrared Spectroscopy to Improve Soil Phosphorus Prediction"
# - Wang et al. (2022), "Mark-Spectra - A convolutional neural network for
#   quantitative spectral analysis overcoming spatial relationships"
# - Li et al. (2023), "SCNet - A deep learning network framework for analyzing
#   near-infrared spectroscopy using short-cut"
