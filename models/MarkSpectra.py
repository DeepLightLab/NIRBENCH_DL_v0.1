"""
Mark-Spectra Model Implementation (TensorFlow / Keras)
Implementation by: D.Passos (dmpassos@uagl.pt)

Paper:
Yueting Wang, Minzan Li, Ronghua Ji, Minjuan Wang, Yao Zhang, Lihua Zheng,
"Mark-Spectra: A convolutional neural network for quantitative spectral analysis overcoming spatial relationships",
Computers and Electronics in Agriculture 192 (2022) 106624.

Core idea:
- A "Mark" layer selects a limited set of discrete characteristic wavelengths (features) before the CNN,
  to mitigate collinearity and reduce the influence of local spatial relationships among adjacent wavelengths.

Practical note:
- The paper computes wavelength importance via OLS and uses per-wavelength R² to rank/select.
  That computation depends on (X_train, y_train). During a Keras forward pass, a layer cannot access y.
  Therefore this implementation provides:
    1) MarkLayer: selects wavelengths given precomputed indices (or precomputed scores)
    2) compute_mark_indices_ols_r2(): helper to compute OLS–R² scores/indices from training data
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Layer,
    MaxPooling1D,
    Reshape,
)


__all__ = [
    "MarkLayer",
    "compute_mark_indices_ols_r2",
    "markspectra_model",
]


@tf.keras.utils.register_keras_serializable(package="NIRBench")
class MarkLayer(Layer):
    """
    Mark layer: selects a subset of wavelengths (spectral positions) from the input spectrum.

    Expected input shape:
      - (batch, L) or (batch, L, C)

    Output shape:
      - (batch, K) or (batch, K, C)

    You can provide either:
      - indices: explicit integer indices (shape [K])
      - scores: per-wavelength importance scores (shape [L]) + top_k to select K best wavelengths

    Ordering:
      - If indices are provided, they are used as-is.
      - If scores are provided, indices are selected by descending score (top_k) and used in that order.
        This matches the paper's description that marked wavelengths are "ordered by importance".
    """

    def __init__(
        self,
        indices: Optional[Sequence[int]] = None,
        scores: Optional[Sequence[float]] = None,
        top_k: Optional[int] = None,
        name: str = "mark",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if indices is None and scores is None:
            raise ValueError("MarkLayer requires either `indices` or `scores`.")
        if indices is not None and scores is not None:
            raise ValueError("Provide only one of `indices` or `scores`, not both.")

        self._indices_np: Optional[np.ndarray] = None
        self._scores_np: Optional[np.ndarray] = None

        if indices is not None:
            idx = np.asarray(indices, dtype=np.int32)
            if idx.ndim != 1:
                raise ValueError("`indices` must be 1D.")
            self._indices_np = idx
            self.top_k = int(idx.shape[0])
        else:
            sc = np.asarray(scores, dtype=np.float32)
            if sc.ndim != 1:
                raise ValueError("`scores` must be 1D.")
            if top_k is None:
                raise ValueError("When using `scores`, you must set `top_k`.")
            self._scores_np = sc
            self.top_k = int(top_k)

        self._indices_tf: Optional[tf.Tensor] = None

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "indices": None if self._indices_np is None else self._indices_np.tolist(),
                "scores": None if self._scores_np is None else self._scores_np.tolist(),
                "top_k": self.top_k,
                "name": self.name,
            }
        )
        return cfg

    def build(self, input_shape):
        # input_shape: (batch, L) or (batch, L, C)
        if len(input_shape) not in (2, 3):
            raise ValueError(f"MarkLayer expects rank-2 or rank-3 input, got shape={input_shape}.")

        L = int(input_shape[1])
        if self._indices_np is not None:
            if np.any(self._indices_np < 0) or np.any(self._indices_np >= L):
                raise ValueError(f"Some indices are out of range for input length L={L}.")
            self._indices_tf = tf.constant(self._indices_np, dtype=tf.int32)
        else:
            if self._scores_np is None:
                raise RuntimeError("Internal error: scores not set.")
            if self._scores_np.shape[0] != L:
                raise ValueError(f"`scores` length ({self._scores_np.shape[0]}) must match input length L={L}.")
            scores_tf = tf.constant(self._scores_np, dtype=tf.float32)
            top = tf.math.top_k(scores_tf, k=self.top_k, sorted=True)
            self._indices_tf = tf.cast(top.indices, tf.int32)

        super().build(input_shape)

    def call(self, inputs, training=None):
        ## CHANGED DP##
        # Avoid using `tf.rank(x)` in a Python `if` (breaks graph building); the rank
        # is known from `build()` for this layer's supported inputs.
        x = inputs
        if x.shape.rank == 2:
            x = tf.expand_dims(x, axis=-1)  # (batch, L, 1)
            y = tf.gather(x, self._indices_tf, axis=1)
            return tf.squeeze(y, axis=-1)  # (batch, K)
        if x.shape.rank == 3:
            return tf.gather(x, self._indices_tf, axis=1)  # (batch, K, C)
        raise ValueError(
            f"MarkLayer expects rank-2 or rank-3 input, got shape={x.shape}."
        )


def compute_mark_indices_ols_r2(
    X_train: np.ndarray,
    y_train: np.ndarray,
    top_k: int,
    *,
    center_y: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-wavelength importance using univariate OLS and rank wavelengths by R².

    For each wavelength j:
      y ≈ a_j * x_j + b_j
      R²_j = 1 - SSE_j / SST

    Returns:
      indices: shape (top_k,), ordered by descending R²
      r2_scores: shape (n_wavelengths,)
    """
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X_train must be 2D: (n_samples, n_wavelengths).")
    if y.shape[0] != X.shape[0]:
        raise ValueError("X_train and y_train must have the same number of samples.")

    n, L = X.shape
    if top_k <= 0 or top_k > L:
        raise ValueError(f"top_k must be in [1, {L}], got {top_k}.")

    y_mean = y.mean()
    if center_y:
        y0 = y - y_mean
        sst = np.sum(y0 ** 2)
    else:
        sst = np.sum((y - y_mean) ** 2)

    if sst == 0.0:
        r2 = np.zeros(L, dtype=np.float64)
        idx = np.arange(top_k, dtype=np.int32)
        return idx, r2.astype(np.float32)

    x_mean = X.mean(axis=0)
    x0 = X - x_mean
    var_x = np.sum(x0 * x0, axis=0)
    cov_xy = np.sum(x0 * (y[:, None] - y_mean), axis=0)

    a = np.zeros(L, dtype=np.float64)
    nonzero = var_x > 0
    a[nonzero] = cov_xy[nonzero] / var_x[nonzero]
    b = y_mean - a * x_mean

    y_hat = X * a[None, :] + b[None, :]
    sse = np.sum((y[:, None] - y_hat) ** 2, axis=0)
    r2 = 1.0 - (sse / sst)

    idx = np.argsort(-r2)[:top_k].astype(np.int32)
    return idx, r2.astype(np.float32)


def markspectra_model(
    input_dim: int,
    *,
    # Mark-layer parameters
    mark_indices: Optional[Sequence[int]] = None,
    mark_scores: Optional[Sequence[float]] = None,
    mark_top_k: Optional[int] = None,
    # CNN parameters (Fig. 5 in the paper)
    conv_filters: Tuple[int, int, int] = (4, 8, 16),
    kernel_sizes: Tuple[int, int, int] = (7, 3, 5),
    strides: Tuple[int, int, int] = (3, 3, 3),
    padding: str = "same",
    pool_sizes: Tuple[int, int] = (2, 2),
    pool_strides: Optional[Tuple[int, int]] = None,
    # Head parameters
    hidden_units: int = 16,
    dropout_rate: float = 0.2,
    # Initializers
    kernel_initializer: Union[str, Initializer] = "glorot_uniform",
    name: str = "MarkSpectra",
) -> Model:
    """
    Build the Mark-Spectra regression model (output: Dense(1, linear)).

    `mark_indices` OR (`mark_scores` + `mark_top_k`) must be provided to enable the Mark layer.
    """
    if input_dim <= 0:
        raise ValueError("input_dim must be a positive integer.")
    if pool_strides is None:
        pool_strides = pool_sizes

    inp = Input(shape=(input_dim,), name="spectrum")
    x = Reshape((input_dim, 1), name="add_channel")(inp)

    if mark_indices is None and mark_scores is None:
        raise ValueError(
            "Mark-Spectra requires Mark selection. Provide `mark_indices` or (`mark_scores` + `mark_top_k`)."
        )

    x = MarkLayer(indices=mark_indices, scores=mark_scores, top_k=mark_top_k, name="Mark")(x)

    # Conv1 -> BN -> ReLU -> Pool1
    x = Conv1D(
        filters=int(conv_filters[0]),
        kernel_size=int(kernel_sizes[0]),
        strides=int(strides[0]),
        padding=padding,
        kernel_initializer=kernel_initializer,
        name="Conv1",
    )(x)
    x = BatchNormalization(name="BN1")(x)
    x = Activation("relu", name="ReLU1")(x)
    x = MaxPooling1D(
        pool_size=int(pool_sizes[0]),
        strides=int(pool_strides[0]),
        padding=padding,
        name="Pooling1",
    )(x)

    # Conv2 -> BN -> ReLU -> Pool2
    x = Conv1D(
        filters=int(conv_filters[1]),
        kernel_size=int(kernel_sizes[1]),
        strides=int(strides[1]),
        padding=padding,
        kernel_initializer=kernel_initializer,
        name="Conv2",
    )(x)
    x = BatchNormalization(name="BN2")(x)
    x = Activation("relu", name="ReLU2")(x)
    x = MaxPooling1D(
        pool_size=int(pool_sizes[1]),
        strides=int(pool_strides[1]),
        padding=padding,
        name="Pooling2",
    )(x)

    # Conv3 -> BN -> ReLU
    x = Conv1D(
        filters=int(conv_filters[2]),
        kernel_size=int(kernel_sizes[2]),
        strides=int(strides[2]),
        padding=padding,
        kernel_initializer=kernel_initializer,
        name="Conv3",
    )(x)
    x = BatchNormalization(name="BN3")(x)
    x = Activation("relu", name="ReLU3")(x)

    # Flatten -> BN -> Dropout -> Dense(F1) -> ReLU -> Dropout
    x = Flatten(name="Flatten")(x)
    x = BatchNormalization(name="BN_flatten")(x)
    x = Dropout(dropout_rate, name="Dropout0")(x)
    x = Dense(hidden_units, kernel_initializer=kernel_initializer, name="F1")(x)
    x = Activation("relu", name="ReLU_F1")(x)
    x = Dropout(dropout_rate, name="Dropout1")(x)

    out = Dense(1, activation="linear", kernel_initializer=kernel_initializer, name="Output")(x)
    return Model(inputs=inp, outputs=out, name=name)


# ---------------------------------------------------------------------------
# Paper hyperparameters (as reported)
# ---------------------------------------------------------------------------
# Optimizer and loss:
# - Adam optimizer; loss = Mean Squared Error (MSE).
#
# Architecture (Mark-Spectra, Fig. 5):
# - Mark layer selects characteristic wavelengths (g) according to importance weights.
# - Conv1 -> Pool1 -> Conv2 -> Pool2 -> Conv3 -> Flatten -> F1 -> Output.
# - Conv filters: (4, 8, 16).
# - Activation: ReLU for conv and fully connected layers.
# - Batch Normalization (BN): after each convolution layer and after Flatten.
#
# Dataset-dependent hyperparameters reported in Table 1:
# - Corn:
#   kernel_sizes = (7, 3, 5)
#   strides      = (3, 3, 3)
#   hidden_units = 16
#   dropout      = 0.2
#   batch_size   = 32
#   learning_rate= 0.01
#   lr_decay     = 0.001
# - Wheat:
#   kernel_sizes = (3, 2, 3)
#   strides      = (3, 2, 2)
#   hidden_units = 32
#   dropout      = 0.1
#   batch_size   = 128
#   learning_rate= 0.01
#   lr_decay     = 0.001
# - Soil:
#   kernel_sizes = (7, 5, 3)
#   strides      = (5, 3, 2)
#   hidden_units = 64
#   dropout      = 0.5
#   batch_size   = 256
#   learning_rate= 0.01
#   lr_decay     = 0.001
#
# Characteristic wavelengths g:
# - Evaluated from 10% to 90% of features (step 10%) across datasets.
