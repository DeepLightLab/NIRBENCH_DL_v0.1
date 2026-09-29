"""
SpectraNet-32 (adapted for regression) - TensorFlow/Keras

Paper:
  Martins et al. (2023)
  "Estimation of soluble solids content and fruit temperature in 'Rocha' pear
   using Vis-NIR spectroscopy and the SpectraNet-32 deep learning architecture"
  Postharvest Biology and Technology 199 (2023) 112281.

This implementation follows Table 2 of the paper (SpectraNet-32 Architecture)
and adapts the final fully-connected head to a single-output regression target,
matching the NIRBENCH benchmark protocol.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="NIRBench")
class ParametricGELU(layers.Layer):
    """
    Parametric GELU with learnable per-channel mean and std.

    The paper's architecture table counts learnable "Mean" and "Std" parameters
    for each GELU activation block. We parameterize std via softplus to keep it
    positive while preserving a trainable scalar per channel.
    """

    def __init__(self, epsilon: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)
        self._sqrt2 = tf.constant(2.0 ** 0.5, dtype=tf.float32)

    def build(self, input_shape):
        if input_shape is None or len(input_shape) < 1:
            raise ValueError("ParametricGELU requires a known input shape.")
        channels = int(input_shape[-1])
        self.mu = self.add_weight(
            name="mu",
            shape=(channels,),
            initializer="zeros",
            trainable=True,
        )
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
    return layers.Activation(tf.nn.gelu, name=name)


def _l2(l2_coeff: float | None):
    if l2_coeff is None or l2_coeff <= 0:
        return None
    return keras.regularizers.l2(float(l2_coeff))


def _conv1d(
    x: tf.Tensor,
    *,
    filters: int,
    kernel_size: int,
    stride: int,
    l2_coeff: float,
    kernel_initializer: str = "he_normal",
    name: str,
) -> tf.Tensor:
    return layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=True,
        kernel_initializer=kernel_initializer,
        bias_initializer="zeros",
        kernel_regularizer=_l2(l2_coeff),
        name=name,
    )(x)


def _bn(x: tf.Tensor, *, name: str) -> tf.Tensor:
    return layers.BatchNormalization(name=name)(x)


def _residual_unit_basic(
    x: tf.Tensor,
    *,
    filters: int,
    kernel_size: int,
    l2_coeff: float,
    use_parametric_gelu: bool,
    name: str,
) -> tf.Tensor:
    shortcut = x

    y = _conv1d(
        x,
        filters=filters,
        kernel_size=kernel_size,
        stride=1,
        l2_coeff=l2_coeff,
        name=f"{name}_conv1",
    )
    y = _bn(y, name=f"{name}_bn1")
    y = _act_layer(use_parametric_gelu, name=f"{name}_gelu1")(y)

    y = _conv1d(
        y,
        filters=filters,
        kernel_size=kernel_size,
        stride=1,
        l2_coeff=l2_coeff,
        name=f"{name}_conv2",
    )
    y = _bn(y, name=f"{name}_bn2")

    y = layers.Add(name=f"{name}_add")([y, shortcut])
    y = _act_layer(use_parametric_gelu, name=f"{name}_gelu_out")(y)
    return y


def _residual_unit_downsample(
    x: tf.Tensor,
    *,
    filters: int,
    kernel_size: int,
    l2_coeff: float,
    use_parametric_gelu: bool,
    name: str,
) -> tf.Tensor:
    y = _conv1d(
        x,
        filters=filters,
        kernel_size=kernel_size,
        stride=2,
        l2_coeff=l2_coeff,
        name=f"{name}_conv1_s2",
    )
    y = _bn(y, name=f"{name}_bn1")
    y = _act_layer(use_parametric_gelu, name=f"{name}_gelu1")(y)

    y = _conv1d(
        y,
        filters=filters,
        kernel_size=kernel_size,
        stride=1,
        l2_coeff=l2_coeff,
        name=f"{name}_conv2",
    )
    y = _bn(y, name=f"{name}_bn2")

    sc = layers.Conv1D(
        filters=filters,
        kernel_size=1,
        strides=2,
        padding="same",
        use_bias=True,
        kernel_initializer="he_normal",
        bias_initializer="zeros",
        kernel_regularizer=_l2(l2_coeff),
        name=f"{name}_skip_conv1x1_s2",
    )(x)
    sc = _bn(sc, name=f"{name}_skip_bn")

    out = layers.Add(name=f"{name}_add")([y, sc])
    out = _act_layer(use_parametric_gelu, name=f"{name}_gelu_out")(out)
    return out


def build_spectranet32(
    input_dim: int,
    *,
    input_channels: int = 1,
    kernel_size: int = 21,
    stem_filters: int = 32,
    stage2_filters: int = 64,
    stage3_filters: int = 128,
    dropout_rate: float = 0.25,
    l2_coeff: float = 0.05,
    use_parametric_gelu: bool = True,
    input_batchnorm: bool = True,
    name: str = "SpectraNet32",
) -> keras.Model:
    """
    Build SpectraNet-32 with a regression head (Dense(1)).

    Table 2 in the paper uses a 2-output fully connected layer (SSC + temperature).
    For NIRBENCH this is adapted to a single scalar regression output.
    """

    if input_dim <= 0:
        raise ValueError("input_dim must be a positive integer.")
    if input_channels <= 0:
        raise ValueError("input_channels must be a positive integer.")

    inputs = keras.Input(shape=(input_dim, input_channels), name="input")
    x = inputs

    if input_batchnorm:
        x = _bn(x, name="bn_input")

    # Stem: Conv(21, 32) -> BN -> GELU
    x = _conv1d(
        x,
        filters=stem_filters,
        kernel_size=kernel_size,
        stride=1,
        l2_coeff=l2_coeff,
        name="stem_conv",
    )
    x = _bn(x, name="stem_bn")
    x = _act_layer(use_parametric_gelu, name="stem_gelu")(x)

    # RU blocks (32-layer configuration in the paper):
    #   Stage 1: one identity RU at 32 ch
    #   Stage 2: one downsampling RU to 64 ch
    #   Stage 3: one downsampling RU to 128 ch
    x = _residual_unit_basic(
        x,
        filters=stem_filters,
        kernel_size=kernel_size,
        l2_coeff=l2_coeff,
        use_parametric_gelu=use_parametric_gelu,
        name="stage1_unit1",
    )
    x = _residual_unit_downsample(
        x,
        filters=stage2_filters,
        kernel_size=kernel_size,
        l2_coeff=l2_coeff,
        use_parametric_gelu=use_parametric_gelu,
        name="stage2_unit1",
    )
    x = _residual_unit_downsample(
        x,
        filters=stage3_filters,
        kernel_size=kernel_size,
        l2_coeff=l2_coeff,
        use_parametric_gelu=use_parametric_gelu,
        name="stage3_unit1",
    )

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dropout(rate=dropout_rate, name="dropout")(x)

    outputs = layers.Dense(
        1,
        activation="linear",
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=_l2(l2_coeff),
        name="regression",
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


# --------------------------------------------------------------------------------------
# Paper hyperparameters / implementation notes (from Martins et al., 2023)
# --------------------------------------------------------------------------------------
#
# Architecture (Table 2):
# - 32 layers total, 702,276 trainable parameters.
# - Input -> BatchNorm (learnable input standardization / data augmentation, Simon et al., 2016)
# - Stem: Conv1D(filters=32, kernel=21, stride=1) + BN + GELU
# - Residual Units (3 RU blocks, reduced from 6 in SpectraNet-53):
#     Stage 1: 1x RU (identity shortcut), Conv(21, 32) + BN + GELU, Conv(21, 32) + BN, Add, GELU
#     Stage 2: 1x RU downsample (Conv stride=2 to 64 + shortcut Conv1x1 stride=2), kernel=21
#     Stage 3: 1x RU downsample (Conv stride=2 to 128 + shortcut Conv1x1 stride=2), kernel=21
# - GlobalAveragePooling1D
# - Dropout(p=0.25)
# - Fully connected output: 2 outputs in SpectraNet2-32 (SSC + temperature); 1 output in SpectraNet1-32 (SSC only)
#   Multi-output (SpectraNet2) always gave better SSC predictions than SSC-only training.
#
# Activation:
# - GELU with learnable μ and σ per channel (ParametricGELU, 1×1×C shape each).
#   Standard GELU is available as a fallback option in this implementation.
#
# Regularization & training (as reported in Section 2.5):
# - L2 regularization (weight decay): λ = 0.05
# - Gradient clipping: global L2-norm threshold of 0.8 (all gradients scaled by 0.8/L2_norm if exceeded)
# - Optimizer: Adam (default parameters)
# - Learning rate: 1e-3 with drop multiplier 0.90 (-10%) at each epoch
# - Epochs: 15 (fixed, no early stopping; training error stabilized by then)
# - Batch size: 128
# - Loss function: MSE (Mean Squared Error)
# - Weight initialization:
#     Conv layers: He initialization (He et al., 2015)
#     FC layer: Glorot initialization (Glorot and Bengio, 2010)
#
# Preprocessing / input (applied BEFORE the network):
# - Absorbance transform: X_A = -log10(X_R + 0.1), where X_R is raw reflectance.
# - QNV-15 (Quantile Normal Variate with 15 quantiles) on both input and output data.
# - Savitzky-Golay smoothing: 1st or 2nd order derivative, 2nd polynomial order, 51-pt window (~35 nm).
# - Feature selection: PLS-based wrapper (BVE-PLS) or PLS VIP filtering (thresholds 0.8, 1.0, 1.2).
# - Spectral range (original): 432.6–1146.74 nm (1024 wl); trimmed to 499.73–1101.83 nm (874 wl).
#
# Under-specifications / practical notes for this TF implementation:
# - The paper used MATLAB R2021a; small numerical differences vs TF/Keras are expected.
# - QNV-15 preprocessing is kept outside the model (dataset pipeline).
# - LR schedule can be implemented with keras.optimizers.schedules or a callback; not embedded here.
# - Original 2-output head adapted to Dense(1, linear) for NIRBENCH single-target regression.
# - Validation: 5-fold external validation (non-overlapping time periods), 30 networks per fold.
# - Hardware reference: AMD Ryzen 9 5900X + Nvidia RTX 2080 Ti; ~30s/network, ~8012 spectra/s inference.
#
# Example compile suggestion (not enforced):
#   model = build_spectranet32(input_dim=874)
#   opt = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=0.8)
#   model.compile(optimizer=opt, loss="mse", metrics=[keras.metrics.RootMeanSquaredError()])
#

