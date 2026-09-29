"""
IPA (Inception for Petroleum Analysis) model factory — TensorFlow/Keras >= 2.20
Implementation by: D.Passos (dmpassos@uagl.pt)

Paper:
F. Haffner et al., "IPA: A deep CNN based on Inception for Petroleum Analysis",
Fuel 379 (2025) 133016.

This file is an *importable* model-factory implementation for benchmarking:
- First argument: input_vector_dimension
- Remaining args: architecture hyperparameters
- Output: Dense(1, activation="linear") (regression)

Consistency check vs authors' provided implementation:
- The uploaded `IPA_architecture.py` defines:
  * a 3-Conv stem (first conv stride=2, kernel=3, no padding -> 'valid')
  * a 4-branch "Module_35x35" with only kernel sizes {1,3}, stride 2 reductions
  * HeNormal init + L2 regularization on conv weights
  * LeakyReLU activations
  * Flatten -> Dropout -> Dense(1)
- This implementation preserves those architectural choices and exposes them as parameters.
"""

from __future__ import annotations

from typing import Optional, Union

import tensorflow as tf
from tensorflow.keras import Model, initializers
from tensorflow.keras.layers import (
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    MaxPooling1D,
)
from tensorflow.keras.regularizers import L2


__all__ = ["ipa_model"]


def _conv1d_lrelu(
    x: tf.Tensor,
    *,
    filters: int,
    kernel_size: int,
    strides: int = 1,
    padding: str = "valid",
    kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
    kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
    name: Optional[str] = None,
) -> tf.Tensor:
    """Conv1D -> LeakyReLU (paper uses LeakyReLU; authors' code uses LeakyReLU)."""
    x = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name=name,
    )(x)
    x = LeakyReLU(name=None if name is None else f"{name}_lrelu")(x)
    return x


def _module_35x35(
    x: tf.Tensor,
    *,
    base_filters: int,
    padding: str,
    kernel_initializer: tf.keras.initializers.Initializer,
    kernel_regularizer: tf.keras.regularizers.Regularizer,
    concat_axis: int = 1,
    name: str = "module_35x35",
) -> tf.Tensor:
    """
    4-branch module matching the authors' `Module_35x35` block.

    IMPORTANT:
    - The authors' code concatenates with `tf.concat(..., axis=1)`, i.e., along the spectral/length axis.
      This is unusual vs standard Inception (which concatenates channels), but it avoids requiring equal
      branch lengths when using 'valid' padding + different strides/pooling.
    - We keep that behavior by default via concat_axis=1.
    """
    # Branch 1: MaxPool(pool=2,stride=2 by default) -> Conv1D(1, stride=2, filters=2*base)
    b1 = MaxPooling1D(pool_size=2, padding=padding, name=f"{name}_b1_pool")(x)
    b1 = _conv1d_lrelu(
        b1,
        filters=base_filters * 2,
        kernel_size=1,
        strides=2,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name=f"{name}_b1_c1",
    )

    # Branch 2: Conv1D(1, stride=2, filters=base) -> Conv1D(3, stride=1, filters=2*base)
    b2 = _conv1d_lrelu(
        x,
        filters=base_filters,
        kernel_size=1,
        strides=2,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name=f"{name}_b2_c1",
    )
    b2 = _conv1d_lrelu(
        b2,
        filters=base_filters * 2,
        kernel_size=3,
        strides=1,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name=f"{name}_b2_c3",
    )

    # Branch 3: Conv1D(1, stride=2, filters=base) -> Conv1D(3, stride=2, filters=2*base) -> Conv1D(3, stride=2, filters=2*base)
    b3 = _conv1d_lrelu(
        x,
        filters=base_filters,
        kernel_size=1,
        strides=2,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name=f"{name}_b3_c1",
    )
    b3 = _conv1d_lrelu(
        b3,
        filters=base_filters * 2,
        kernel_size=3,
        strides=2,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name=f"{name}_b3_c3a",
    )
    b3 = _conv1d_lrelu(
        b3,
        filters=base_filters * 2,
        kernel_size=3,
        strides=2,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name=f"{name}_b3_c3b",
    )

    # Branch 4: Conv1D(1, stride=2, filters=2*base)
    b4 = _conv1d_lrelu(
        x,
        filters=base_filters * 2,
        kernel_size=1,
        strides=2,
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name=f"{name}_b4_c1",
    )

    # Concatenate along spectral axis by default (to mirror authors' implementation)
    out = Concatenate(axis=concat_axis, name=f"{name}_concat")([b1, b2, b3, b4])
    return out


def ipa_model(
    input_vector_dimension: int,
    *,
    input_channels: int = 1,
    # Stem hyperparameters
    stem_filters: int = 16,
    stem_kernel_size: int = 3,
    stem_first_stride: int = 2,
    stem_n_convs: int = 3,
    # Multi-branch module hyperparameters
    module_base_filters: int = 32,
    module_concat_axis: int = 1,
    # Regularization / init / dropout
    dropout_rate: float = 0.2,
    l2_regularization: float = 1e-3,
    he_seed: int = 1,
    # Convolution padding (paper: no padding -> 'valid')
    padding: str = "valid",
    name: str = "IPA",
) -> Model:
    """
    Build IPA for regression.

    Parameters
    ----------
    input_vector_dimension : int
        Number of spectral variables (length L).
    input_channels : int
        Usually 1 for NIR spectra shaped (L, 1).
    stem_filters, stem_kernel_size, stem_first_stride, stem_n_convs :
        Control the 3-convolution "stem".
    module_base_filters, module_concat_axis :
        Control the 4-branch Inception-like module.
        NOTE: default concat axis=1 to mirror authors' reference implementation.
    dropout_rate : float
        Dropout applied after Flatten, before final regressor.
    l2_regularization : float
        L2 applied to Conv1D weights (kernel_regularizer).
    he_seed : int
        Seed for HeNormal initializer.
    padding : str
        Paper states convolutions use no padding => 'valid' (default).
    """
    if input_vector_dimension <= 0:
        raise ValueError("input_vector_dimension must be a positive integer.")
    if stem_n_convs < 1:
        raise ValueError("stem_n_convs must be >= 1.")

    kernel_init = initializers.HeNormal(seed=he_seed)
    kernel_reg = L2(l2_regularization)

    inputs = Input(shape=(input_vector_dimension, input_channels), name="input")

    # --- Stem: 3 convs, first with stride=2, all kernel=3, no padding ('valid') ---
    x = _conv1d_lrelu(
        inputs,
        filters=stem_filters,
        kernel_size=stem_kernel_size,
        strides=stem_first_stride,
        padding=padding,
        kernel_initializer=kernel_init,
        kernel_regularizer=kernel_reg,
        name="stem_conv1",
    )
    # Remaining stem convs: stride=1
    for i in range(2, stem_n_convs + 1):
        x = _conv1d_lrelu(
            x,
            filters=stem_filters,
            kernel_size=stem_kernel_size,
            strides=1,
            padding=padding,
            kernel_initializer=kernel_init,
            kernel_regularizer=kernel_reg,
            name=f"stem_conv{i}",
        )

    # --- Inception-like module (4 branches) ---
    x = _module_35x35(
        x,
        base_filters=module_base_filters,
        padding=padding,
        kernel_initializer=kernel_init,
        kernel_regularizer=kernel_reg,
        concat_axis=module_concat_axis,
        name="module_35x35",
    )

    # --- Regressor head ---
    x = Flatten(name="flatten")(x)
    x = Dropout(rate=dropout_rate, name="dropout")(x)
    outputs = Dense(1, activation="linear", name="output")(x)

    return Model(inputs=inputs, outputs=outputs, name=name)


# =============================================================================
# Paper hyperparameters / training setup (as reported)
# =============================================================================
#
# Architecture summary (Fig. 3 + text):
# - Stem: 3 Conv1D layers at input.
# - Multi-branch module: 4 parallel paths inspired by Inception (Inception-V2 35x35 module).
# - Kernels only of size 1 and 3; no padding on convolutions (=> 'valid').
# - Activation: LeakyReLU; initialization: He (HeNormal).
# - Total conv layers reported: 10; total parameters ~215k.
#
# Training / optimization (Section around Eq. (4)):
# - Optimizer: Adam.
# - Initial learning rate tuned in [1e-2, 1e-5].
# - LR schedule: Exponential Decay with:
#       lr_decay = 1e-3
#       lr_decay_steps = 1e4
# - Epochs: 500; early stopping patience: 50 epochs.
# - Batch size: 16.
# - Dropout: single Dropout layer with 20% rate in regression part.
# - Loss: MAE + lambda * ||w||_2^2 (L2 regularization on conv weights),
#         with lambda tuned in [1e-2, 1e-5].
#
# Notes for benchmark usage:
# - This file returns an *uncompiled* model (model factory pattern).
# - To mirror the paper loss, compile with MAE and rely on kernel_regularizer=L2(...)
#   contributing to the total loss:
#       model.compile(optimizer=..., loss="mae", metrics=[...])
# - To make it unform with other models, use MSE as metric the loss
# =============================================================================
