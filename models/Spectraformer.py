"""
Spectraformer — TensorFlow/Keras >= 2.20 (adapted for regression)
Implementation by: D.Passos (dmpassos@uagl.pt)

Paper:
Zhuo Chen, Rigui Zhou, Pengju Ren, "Spectraformer: deep learning model for grain spectral qualitative analysis based on transformer structure",
RSC Advances (2024) 14, 8053–8066. DOI: 10.1039/d3ra07708j

NIR-Bench adaptation:
- The paper defines Spectraformer for multi-class classification (cross-entropy).
- Here we adapt the head for regression by using Dense(1, activation="linear") as the final layer.

Important:
- The paper provides a clear macro-architecture (Fig. 3) and some hyperparameters (channels, strides, heads),
  but several implementation details are not fully specified (e.g., positional encoding, exact FC widths, dropout).
  Those are exposed as hyperparameters and documented at the end of this file.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras import Model, layers

__all__ = ["build_spectraformer"]


def _sinusoidal_positional_encoding(seq_len: tf.Tensor, d_model: int, dtype: tf.DType) -> tf.Tensor:
    """
    Classic sinusoidal positional encoding (Vaswani et al., 2017), computed dynamically for seq_len.
    Returns a tensor of shape (1, seq_len, d_model).
    """
    seq_len = tf.cast(seq_len, tf.int32)

    position = tf.cast(tf.range(seq_len)[:, tf.newaxis], dtype)  # (L,1)
    i = tf.cast(tf.range(d_model)[tf.newaxis, :], dtype)         # (1,D)

    angle_rates = 1.0 / tf.pow(10000.0, (2.0 * tf.floor(i / 2.0)) / tf.cast(d_model, dtype))
    angles = position * angle_rates  # (L,D)

    sines = tf.sin(angles[:, 0::2])
    cosines = tf.cos(angles[:, 1::2])

    if d_model % 2 == 0:
        pe = tf.reshape(tf.stack([sines, cosines], axis=-1), [seq_len, d_model])
    else:
        # last dim has no cosine pair
        cos_pad = tf.pad(cosines, [[0, 0], [0, 1]])
        pe = tf.reshape(tf.stack([sines, cos_pad], axis=-1), [seq_len, d_model])

    return pe[tf.newaxis, :, :]  # (1,L,D)


@tf.keras.utils.register_keras_serializable(package="NIRBench")
class TransformerBlock(layers.Layer):
    """
    Minimal transformer encoder block:
      x -> (+pos enc) -> LN -> MHA -> Dropout -> Residual
        -> LN -> MLP -> Dropout -> Residual
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 2,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        use_positional_encoding: Union[bool, str] = "sinusoidal",
        name: str = "TransformerBlock",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.attn_dropout = float(attn_dropout)
        self.mlp_dropout = float(mlp_dropout)
        self.use_positional_encoding = use_positional_encoding

        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name="ln1")
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.attn_dropout,
            name="mha",
        )
        self.drop_attn = layers.Dropout(self.attn_dropout, name="drop_attn")

        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name="ln2")
        mlp_hidden = int(round(self.embed_dim * self.mlp_ratio))
        self.mlp_dense1 = layers.Dense(mlp_hidden, activation="gelu", name="mlp_dense1")
        self.drop_mlp1 = layers.Dropout(self.mlp_dropout, name="drop_mlp1")
        self.mlp_dense2 = layers.Dense(self.embed_dim, activation=None, name="mlp_dense2")
        self.drop_mlp2 = layers.Dropout(self.mlp_dropout, name="drop_mlp2")

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        if self.use_positional_encoding:
            use_pe = (
                (isinstance(self.use_positional_encoding, str) and self.use_positional_encoding.lower() == "sinusoidal")
                or (self.use_positional_encoding is True)
            )
            if use_pe:
                x = x + _sinusoidal_positional_encoding(tf.shape(x)[1], self.embed_dim, x.dtype)

        y = self.ln1(x)
        y = self.mha(y, y, training=training)
        y = self.drop_attn(y, training=training)
        x = x + y

        y = self.ln2(x)
        y = self.mlp_dense1(y)
        y = self.drop_mlp1(y, training=training)
        y = self.mlp_dense2(y)
        y = self.drop_mlp2(y, training=training)
        x = x + y
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            dict(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_dropout=self.attn_dropout,
                mlp_dropout=self.mlp_dropout,
                use_positional_encoding=self.use_positional_encoding,
            )
        )
        return cfg


def build_spectraformer(
    input_vector_dimension: int,
    *,
    input_channels: int = 1,
    # Convolutional trunk (paper: 4 blocks; each block = conv(k=3) + conv(k=1), BN+ReLU after each conv)
    num_conv_blocks: int = 4,
    initial_filters: int = 16,
    kernel_size_3: int = 3,
    kernel_size_1: int = 1,
    conv_stride: int = 2,
    conv_padding: str = "same",
    # Transformer placement and params (paper: transformer after the first conv block; heads=2)
    transformer_after_block: int = 1,
    transformer_num_heads: int = 2,
    transformer_mlp_ratio: float = 4.0,
    transformer_attn_dropout: float = 0.0,
    transformer_mlp_dropout: float = 0.0,
    transformer_use_positional_encoding: Union[bool, str] = "sinusoidal",
    # Head (paper: 2 fully connected layers; widths not specified)
    dense_units: Tuple[int, int] = (256, 128),
    dense_activation: str = "relu",
    dense_dropout: float = 0.5,
    name: str = "Spectraformer",
) -> Model:
    """
    Build Spectraformer adapted for regression.

    Parameters
    ----------
    input_vector_dimension : int
        Number of spectral variables (length L).
    input_channels : int
        Usually 1 for NIR spectra shaped (L,1).
    num_conv_blocks : int
        Paper uses 4 convolutional blocks (k=3 then k=1 per block).
    initial_filters : int
        Paper sets the first conv to 16 channels; subsequent blocks double channels.
    conv_stride : int
        Paper states stride=2 for all convolution layers.
    conv_padding : str
        Paper reports padding=1 (PyTorch). In TF/Keras we default to 'same' as a practical approximation.
    transformer_after_block : int
        Paper reports best placement after the first conv block (value=1).
    dense_units : (int, int)
        Two FC layer widths. Not specified in paper; exposed for tuning.
    dense_dropout : float
        Dropout in the head (not specified in paper).
    """
    if input_vector_dimension <= 0:
        raise ValueError("input_vector_dimension must be a positive integer.")
    if input_channels <= 0:
        raise ValueError("input_channels must be a positive integer.")
    if num_conv_blocks <= 0:
        raise ValueError("num_conv_blocks must be positive.")
    if transformer_after_block < 0 or transformer_after_block > num_conv_blocks:
        raise ValueError("transformer_after_block must be in [0, num_conv_blocks].")
    if conv_padding not in ("same", "valid", "causal"):
        raise ValueError("conv_padding must be one of {'same','valid','causal'} for Keras Conv1D.")
    if len(dense_units) != 2:
        raise ValueError("dense_units must be a tuple of length 2, e.g., (256, 128).")

    inputs = layers.Input(shape=(input_vector_dimension, input_channels), name="input")
    x = inputs

    filters = int(initial_filters)

    for b in range(1, num_conv_blocks + 1):
        # k=3 convolution
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size_3,
            strides=conv_stride,
            padding=conv_padding,
            use_bias=False,
            kernel_initializer="he_normal",
            name=f"block{b}_conv3",
        )(x)
        x = layers.BatchNormalization(name=f"block{b}_bn3")(x)
        x = layers.ReLU(name=f"block{b}_relu3")(x)

        # k=1 convolution
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size_1,
            strides=conv_stride,
            padding=conv_padding,
            use_bias=False,
            kernel_initializer="he_normal",
            name=f"block{b}_conv1",
        )(x)
        x = layers.BatchNormalization(name=f"block{b}_bn1")(x)
        x = layers.ReLU(name=f"block{b}_relu1")(x)

        if b == transformer_after_block:
            x = TransformerBlock(
                embed_dim=filters,
                num_heads=transformer_num_heads,
                mlp_ratio=transformer_mlp_ratio,
                attn_dropout=transformer_attn_dropout,
                mlp_dropout=transformer_mlp_dropout,
                use_positional_encoding=transformer_use_positional_encoding,
                name="transformer",
            )(x)

        filters *= 2

    x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(dense_units[0], activation=dense_activation, name="fc1")(x)
    x = layers.Dropout(dense_dropout, name="drop1")(x)
    x = layers.Dense(dense_units[1], activation=dense_activation, name="fc2")(x)
    x = layers.Dropout(dense_dropout, name="drop2")(x)

    outputs = layers.Dense(1, activation="linear", name="output")(x)

    return Model(inputs=inputs, outputs=outputs, name=name)


# =============================================================================
# Paper hyperparameters / training setup (original classification setting)
# =============================================================================
#
# From the paper (Fig. 3 and Section 3.1):
# - Macro-architecture:
#   * Transformer module inserted after the first convolutional block.
#   * 4 × Conv1D(kernel=3) and 4 × Conv1D(kernel=1), organized as 4 blocks with BN+ReLU after each conv.
#   * Two fully-connected layers after flattening.
# - Channels:
#   * First conv uses 16 channels; channel count doubles in later stages.
# - Convolution stride and padding:
#   * Stride is set to 2 across convolutional layers; padding is reported as 1 (PyTorch-style).
#   * TF/Keras does not expose integer padding directly in Conv1D; we approximate with padding='same'.
# - Transformer:
#   * Dual-head self-attention (num_heads=2).
#   * "A transformer block" described as attention mechanisms + MLP.
# - Training (classification):
#   * Loss: cross-entropy
#   * Optimizer: SGD with learning rate 1e-4
#   * Epochs: 200
#
# Missing / under-specified details in the paper (implemented here as tunable defaults):
# - Positional encoding (not described): default sinusoidal
# - Transformer normalization scheme (pre-norm vs post-norm): default pre-norm
# - Transformer MLP hidden width and dropout rates
# - Exact FC layer widths (dense_units) and any dropout usage in the head
#
# Benchmark adaptation:
# - Replace classification head with Dense(1, linear).
# =============================================================================
