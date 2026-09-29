"""
SpectraNet-53 (adapted for regression) - TensorFlow/Keras 2.20
Implementation by: D.Passos (dmpassos@uagl.pt)

Paper: "SpectraNet–53: A deep residual learning architecture for predicting soluble solids content with VIS–NIR spectroscopy"
J.A. Martins et al., Computers and Electronics in Agriculture 197 (2022) 106945.

This module exposes a single importable factory function:
    build_spectranet53(input_dim, ...)

Design goals for NIR Bench:
- First hyperparameter is input_dim (number of wavelengths).
- Model is for regression: last layer is always Dense(1, linear).
- Self-contained and importable.

Notes:
- The original paper describes training in MATLAB and reports a 6-output variant.
  Here we implement the core SpectraNet-53 backbone and adapt the head to regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="NIRBench")
class ParametricGELU(layers.Layer):
    """
    Parametric GELU as described in the paper (learnable mean and std per channel):

        GELU(x | μ, σ) = x * CDF(x | μ, σ)

    where CDF is the Gaussian cumulative distribution function.
    We parameterize σ via softplus to keep it positive.

    This layer expects channels-last tensors: (..., channels).
    """
    def __init__(self, epsilon: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)
        self._sqrt2 = tf.constant(2.0 ** 0.5, dtype=tf.float32)

    def build(self, input_shape):
        if input_shape is None or len(input_shape) < 1:
            raise ValueError("ParametricGELU requires a known input shape.")
        channels = int(input_shape[-1])
        # μ initialized at 0
        self.mu = self.add_weight(
            name="mu",
            shape=(channels,),
            initializer="zeros",
            trainable=True,
        )
        # raw σ initialized so that softplus(raw_sigma) ~ 1
        self.raw_sigma = self.add_weight(
            name="raw_sigma",
            shape=(channels,),
            initializer=keras.initializers.Constant(0.541324854612918),  # softplus^-1(1.0)
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        x = tf.convert_to_tensor(x)
        mu = tf.reshape(self.mu, (1, 1, -1))
        sigma = tf.reshape(tf.nn.softplus(self.raw_sigma) + self.epsilon, (1, 1, -1))
        z = (x - mu) / (sigma * self._sqrt2)
        cdf = 0.5 * (1.0 + tf.math.erf(z))
        return x * cdf

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"epsilon": self.epsilon})
        return cfg


def _act_layer(use_parametric_gelu: bool, name: Optional[str] = None) -> layers.Layer:
    if use_parametric_gelu:
        return ParametricGELU(name=name)
    # Fallback: standard GELU (non-parametric).
    return layers.Activation(tf.nn.gelu, name=name)


def _conv1d(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    stride: int,
    l2_coeff: float,
    name: str,
) -> tf.Tensor:
    return layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=True,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        kernel_regularizer=keras.regularizers.l2(l2_coeff) if l2_coeff and l2_coeff > 0 else None,
        name=name,
    )(x)


def _bn(x: tf.Tensor, name: str) -> tf.Tensor:
    # Default BN params are fine; paper used BN throughout (and also after input).
    return layers.BatchNormalization(name=name)(x)


def _residual_unit_basic(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    l2_coeff: float,
    use_parametric_gelu: bool,
    name: str,
) -> tf.Tensor:
    """
    Basic RU with identity shortcut:
      Conv -> BN -> GELU -> Conv -> BN -> Add(shortcut) -> GELU
    """
    shortcut = x

    y = _conv1d(x, filters, kernel_size, stride=1, l2_coeff=l2_coeff, name=f"{name}_conv1")
    y = _bn(y, name=f"{name}_bn1")
    y = _act_layer(use_parametric_gelu, name=f"{name}_gelu1")(y)

    y = _conv1d(y, filters, kernel_size, stride=1, l2_coeff=l2_coeff, name=f"{name}_conv2")
    y = _bn(y, name=f"{name}_bn2")

    y = layers.Add(name=f"{name}_add")([y, shortcut])
    y = _act_layer(use_parametric_gelu, name=f"{name}_gelu_out")(y)
    return y


def _residual_unit_downsample(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    l2_coeff: float,
    use_parametric_gelu: bool,
    name: str,
) -> tf.Tensor:
    """
    Downsampling RU with stride-2 in the main path and 1x1 stride-2 shortcut:
      Conv(stride=2) -> BN -> GELU -> Conv -> BN
      Shortcut: Conv1x1(stride=2) -> BN
      Add -> GELU
    """
    # Main
    y = _conv1d(x, filters, kernel_size, stride=2, l2_coeff=l2_coeff, name=f"{name}_conv1_s2")
    y = _bn(y, name=f"{name}_bn1")
    y = _act_layer(use_parametric_gelu, name=f"{name}_gelu1")(y)

    y = _conv1d(y, filters, kernel_size, stride=1, l2_coeff=l2_coeff, name=f"{name}_conv2")
    y = _bn(y, name=f"{name}_bn2")

    # Shortcut
    sc = layers.Conv1D(
        filters=filters,
        kernel_size=1,
        strides=2,
        padding="same",
        use_bias=True,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        kernel_regularizer=keras.regularizers.l2(l2_coeff) if l2_coeff and l2_coeff > 0 else None,
        name=f"{name}_skip_conv1x1_s2",
    )(x)
    sc = _bn(sc, name=f"{name}_skip_bn")

    out = layers.Add(name=f"{name}_add")([y, sc])
    out = _act_layer(use_parametric_gelu, name=f"{name}_gelu_out")(out)
    return out


def build_spectranet53(
    input_dim: int,
    *,
    input_channels: int = 1,
    kernel_size: int = 17,
    stem_filters: int = 32,
    stage2_filters: int = 64,
    stage3_filters: int = 128,
    dropout_rate: float = 0.20,
    l2_coeff: float = 0.05,
    use_parametric_gelu: bool = True,
    input_batchnorm: bool = True,
    name: str = "SpectraNet53",
) -> keras.Model:
    """
    Build SpectraNet-53 backbone with a regression head (Dense(1, linear)).

    Parameters
    ----------
    input_dim:
        Number of spectral points (wavelengths).
    input_channels:
        Channels per wavelength (typically 1).
    kernel_size:
        Convolution kernel size. Paper uses 17 across all conv layers.
    stem_filters, stage2_filters, stage3_filters:
        Filters per stage (paper: 32, 64, 128).
    dropout_rate:
        Dropout after global average pooling (paper: 0.20).
    l2_coeff:
        L2 weight decay coefficient (paper: 0.05).
    use_parametric_gelu:
        If True, uses ParametricGELU (learnable μ, σ per channel). Otherwise uses tf.nn.gelu.
    input_batchnorm:
        If True, applies BatchNorm directly after input as in the paper.
    name:
        Keras model name.

    Returns
    -------
    tf.keras.Model
        A model with output shape (None, 1).
    """
    if input_dim <= 0:
        raise ValueError("input_dim must be a positive integer.")
    if input_channels <= 0:
        raise ValueError("input_channels must be a positive integer.")

    inputs = keras.Input(shape=(input_dim, input_channels), name="input")

    x = inputs
    if input_batchnorm:
        x = _bn(x, name="bn_input")

    # Stem (L2 in Table 4): Conv(17, 32) -> BN -> GELU
    x = _conv1d(x, stem_filters, kernel_size, stride=1, l2_coeff=l2_coeff, name="stem_conv")
    x = _bn(x, name="stem_bn")
    x = _act_layer(use_parametric_gelu, name="stem_gelu")(x)

    # Stage 1: 2 residual units, identity shortcuts (S1U1, S1U2)
    x = _residual_unit_basic(
        x,
        filters=stem_filters,
        kernel_size=kernel_size,
        l2_coeff=l2_coeff,
        use_parametric_gelu=use_parametric_gelu,
        name="stage1_unit1",
    )
    x = _residual_unit_basic(
        x,
        filters=stem_filters,
        kernel_size=kernel_size,
        l2_coeff=l2_coeff,
        use_parametric_gelu=use_parametric_gelu,
        name="stage1_unit2",
    )

    # Stage 2: downsample RU + 1 RU (S2U1 uses stride-2 + 1x1 shortcut, then S2U2 identity)
    x = _residual_unit_downsample(
        x,
        filters=stage2_filters,
        kernel_size=kernel_size,
        l2_coeff=l2_coeff,
        use_parametric_gelu=use_parametric_gelu,
        name="stage2_unit1",
    )
    x = _residual_unit_basic(
        x,
        filters=stage2_filters,
        kernel_size=kernel_size,
        l2_coeff=l2_coeff,
        use_parametric_gelu=use_parametric_gelu,
        name="stage2_unit2",
    )

    # Stage 3: downsample RU + 1 RU (S3U1 stride-2 + 1x1 shortcut, then S3U2 identity)
    x = _residual_unit_downsample(
        x,
        filters=stage3_filters,
        kernel_size=kernel_size,
        l2_coeff=l2_coeff,
        use_parametric_gelu=use_parametric_gelu,
        name="stage3_unit1",
    )
    x = _residual_unit_basic(
        x,
        filters=stage3_filters,
        kernel_size=kernel_size,
        l2_coeff=l2_coeff,
        use_parametric_gelu=use_parametric_gelu,
        name="stage3_unit2",
    )

    # Head (paper: GAP -> Dropout(0.2) -> FC(6))
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dropout(rate=dropout_rate, name="dropout")(x)

    # Regression output (benchmark requirement): Dense(1, linear)
    outputs = layers.Dense(
        1,
        activation="linear",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=keras.regularizers.l2(l2_coeff) if l2_coeff and l2_coeff > 0 else None,
        name="regression",
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


# --------------------------------------------------------------------------------------
# Paper hyperparameters / implementation notes (from Martins et al., 2022)
# --------------------------------------------------------------------------------------
#
# Architecture (Table 4):
# - Input -> BatchNorm (after input, uncommon but used here)
# - Stem: Conv1D(filters=32, kernel=17, stride=1) + BN + GELU
# - Residual Units:
#     Stage 1: 2x RU (identity shortcut), each with Conv(17, 32) + BN + GELU, Conv(17, 32) + BN, Add, GELU
#     Stage 2: RU downsample (Conv stride=2 to 64 + shortcut Conv1x1 stride=2) + RU (identity), both with kernel=17
#     Stage 3: RU downsample (Conv stride=2 to 128 + shortcut Conv1x1 stride=2) + RU (identity), both with kernel=17
# - GlobalAveragePooling1D
# - Dropout(p=0.20)
# - Fully connected output: 6 outputs in SpectraNet6-53 (SSC + 5 IQAs); 1 output in SpectraNet-53 (SSC only)
#
# Activation:
# - GELU with learnable μ and σ per channel (ParametricGELU). Standard GELU is a fallback option here.
#
# Regularization & training (as reported):
# - L2 regularization (weight decay): λ = 0.05
# - Gradient clipping: per-parameter L2-norm clipped to 0.5
# - Optimizer: Adam (default params)
# - Learning rate: 1e-4 with drop multiplier 0.7 each epoch
# - Epochs: 10 (early stop point in the paper's main experiments)
# - Batch size: 32
# - Input scaling: QNV-5 (Quantile Normal Variate with 5 quantiles) applied BEFORE the network,
#   plus BN-after-input used as an additional stabilization/augmentation mechanism.
#
# Under-specifications / practical notes for this TF implementation:
# - The paper's QNV-5 is a preprocessing transform; we keep it outside the model (dataset pipeline).
# - The paper used MATLAB; small numerical differences vs TF/Keras are expected.
# - LR schedule can be implemented with keras.optimizers.schedules or a callback; not embedded here.
#
# Example compile suggestion (not enforced):
#   model = build_spectranet53(input_dim=1421)
#   opt = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=0.5)
#   model.compile(optimizer=opt, loss="mse", metrics=[keras.metrics.RootMeanSquaredError()])
#
