"""
SCNet: short-cut concatenate convolutional network for 1D spectra regression.
Implementation by: D.Passos (dmpassos@uagl.pt)

Paper:
Z. Li et al., "SCNet: A deep learning network framework for analyzing near-infrared spectroscopy using short-cut",
Infrared Physics & Technology 132 (2023) 104731.

This implementation is adapted for the Spectral CNN benchmark:
- Functional Keras model factory function.
- First hyperparameter is input_dim (number of spectral points).
- Always returns a regression model with a final Dense(1, linear) output.

TensorFlow / Keras: tested with tensorflow.keras >= 2.16 (target 2.20).
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Optional

import tensorflow as tf
from tensorflow.keras import layers


def build_scnet(
    input_dim: int,
    *,
    conv_filters: Sequence[int] = (8, 8, 8),
    kernel_sizes: Sequence[int] = (7, 5, 3),
    activation: str = "relu",
    use_batchnorm: bool = True,
    branch_pool: str = "gap",
    dense_units: Sequence[int] = (267,),
    dense_activation: str = "relu",
    dropout_rate: float = 0.0,
    l2: float = 0.0,
    name: str = "SCNet",
) -> tf.keras.Model:
    """
    Build SCNet (regression version).

    Parameters
    ----------
    input_dim:
        Number of wavelengths / spectral points (L). Input shape will be (L, 1).
    conv_filters:
        Number of filters for each sequential Conv1D layer in the feature extractor.
        The paper uses a "multiscale convolutional layers" design; we expose filters per stage.
    kernel_sizes:
        Kernel size for each Conv1D layer. Different kernel sizes implement the multiscale aspect.
    activation:
        Nonlinearity applied after each (Conv -> BN) stage.
    use_batchnorm:
        Whether to use BatchNormalization after convolutions.
    branch_pool:
        How to convert each tapped feature map to a vector before concatenation:
        - "gap": GlobalAveragePooling1D
        - "gmp": GlobalMaxPooling1D
        - "flatten": Flatten (keeps full temporal resolution; more parameters)
    dense_units:
        Units of the dense layers after concatenation (the "regression part").
        In the paper, the neuron number of the last linear layer was tuned with grid-search (267).
    dense_activation:
        Activation for the dense layers in the regression head (excluding final output).
    dropout_rate:
        Optional dropout applied after each dense layer (0 disables).
    l2:
        L2 weight decay for Conv/Dense kernels (0 disables).
    name:
        Model name.

    Returns
    -------
    tf.keras.Model
        Keras model mapping (batch, input_dim, 1) -> (batch, 1).
    """
    if input_dim <= 0:
        raise ValueError(f"input_dim must be > 0, got {input_dim}.")
    if len(conv_filters) != len(kernel_sizes):
        raise ValueError(
            f"conv_filters and kernel_sizes must have the same length; "
            f"got {len(conv_filters)} and {len(kernel_sizes)}."
        )
    if branch_pool not in {"gap", "gmp", "flatten"}:
        raise ValueError('branch_pool must be one of {"gap","gmp","flatten"}.')

    reg = tf.keras.regularizers.L2(l2) if l2 and l2 > 0 else None

    inp = layers.Input(shape=(input_dim, 1), name="spectra")

    x = inp
    taps = []

    # Feature extractor: sequential Conv blocks; we "tap" each stage output and
    # concatenate them later (short-cut concatenate).
    for i, (f, k) in enumerate(zip(conv_filters, kernel_sizes), start=1):
        x = layers.Conv1D(
            filters=int(f),
            kernel_size=int(k),
            padding="same",
            use_bias=not use_batchnorm,
            kernel_regularizer=reg,
            name=f"conv{i}",
        )(x)
        if use_batchnorm:
            x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation(activation, name=f"act{i}")(x)
        taps.append(x)

    # Convert each tapped feature map to a vector.
    if branch_pool == "gap":
        pool_layer = layers.GlobalAveragePooling1D
        branch_vecs = [pool_layer(name=f"gap{i}")(t) for i, t in enumerate(taps, start=1)]
    elif branch_pool == "gmp":
        pool_layer = layers.GlobalMaxPooling1D
        branch_vecs = [pool_layer(name=f"gmp{i}")(t) for i, t in enumerate(taps, start=1)]
    else:  # flatten
        branch_vecs = [layers.Flatten(name=f"flat{i}")(t) for i, t in enumerate(taps, start=1)]

    feat = layers.Concatenate(name="shortcut_concat")(branch_vecs)

    # Regression head
    y = feat
    for j, units in enumerate(dense_units, start=1):
        y = layers.Dense(
            int(units),
            activation=dense_activation,
            kernel_regularizer=reg,
            name=f"dense{j}",
        )(y)
        if dropout_rate and dropout_rate > 0:
            y = layers.Dropout(float(dropout_rate), name=f"dropout{j}")(y)

    out = layers.Dense(1, activation="linear", name="y")(y)

    return tf.keras.Model(inputs=inp, outputs=out, name=name)


# -----------------------------------------------------------------------------
# Hyperparameters reported in the paper (for reference)
# -----------------------------------------------------------------------------
# - Task: NIR spectroscopy regression. The paper evaluates SCNet on four public datasets
#   (Corn Moisture, Marzipan Moisture/Sugar, Soil Organic Matter, Mango Dry Matter).
# - Key architectural idea: replace residual "addition" shortcuts with *concatenation*
#   of multi-level features, connected to the feature vector (not to subsequent conv layers).
# - Training protocol (as reported for their main comparison experiment):
#     * 5-fold cross-validation
#     * batch size: 128
#     * learning rate: 0.001
#     * optimizer: Adam
#     * epochs: 20000 (with Early Stopping)
# - Pre-processing (as described):
#     * Savitzky–Golay smoothing (quadratic polynomial), window=17
#     * first derivative (baseline drift correction)
#     * min–max normalization of inputs; targets also scaled to [0, 1]
#     * paper notes sigmoid activation on the output layer after scaling; in this benchmark
#       we enforce a linear output Dense(1) to keep a consistent regression interface.
# - The neuron number of the last linear layer of SCNet was tuned by grid-search on Mango
#   and set to 267 (search range 2..500, step 1).
