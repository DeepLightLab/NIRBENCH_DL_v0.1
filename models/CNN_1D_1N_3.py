"""
1D-CNN_1N_3 baseline model factory for NIRBENCH.

Shallow 1D CNN with one narrow convolutional filter followed by a dense
regression head.
"""

from __future__ import annotations

from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input


__all__ = ["build_cnn_1d_1n_3"]


def build_cnn_1d_1n_3(
    input_vector_dimension: int,
    *,
    input_channels: int = 1,
    l2_coeff: float = 1e-3,
    name: str = "1D-CNN_1N_3",
) -> Model:
    """
    Build the 1D-CNN_1N_3 regression baseline.

    Architecture:
      Input -> Conv1D(1, 5) -> Flatten -> Dense(128) -> Dense(64) -> Dense(16) -> Dense(1)
    """
    if input_vector_dimension <= 0:
        raise ValueError("input_vector_dimension must be a positive integer.")
    if input_channels <= 0:
        raise ValueError("input_channels must be a positive integer.")

    reg = regularizers.l2(float(l2_coeff))

    inputs = Input(shape=(input_vector_dimension, input_channels), name="input")
    x = Conv1D(
        filters=1,
        kernel_size=5,
        strides=1,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=reg,
        name="conv1",
    )(inputs)
    x = Flatten(name="flatten")(x)
    x = Dense(
        128,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=reg,
        name="dense1",
    )(x)
    x = Dense(
        64,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=reg,
        name="dense2",
    )(x)
    x = Dense(
        16,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=reg,
        name="dense3",
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
# - This is the narrow-filter variant of the same shallow single-convolution
#   baseline, emphasizing local spectral motifs instead of a wide receptive
#   field.
#
# References:
# - Acquarelli et al. (2017), "Convolutional neural networks for vibrational
#   spectroscopic data analysis"
# - Cui and Fearn (2018), "Modern practical convolutional neural networks for
#   multivariate regression - Applications to NIR calibration"
# - Luo et al. (2024), "Principles and applications of CNNs for spectral
#   analysis in food quality evaluation"
