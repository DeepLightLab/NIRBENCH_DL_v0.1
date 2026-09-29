"""
SpectraTr (Transformer-for-NIR) - TensorFlow/Keras implementation (v2.20 compatible)
Implementation by: D.Passos (dmpassos@uagl.pt)

Paper:
  P. Fu et al., "SpectraTr: A novel deep learning model for qualitative analysis of drug spectroscopy",
  Journal of Innovative Optical Health Sciences, 2022.

Notes:
- The original paper proposes SpectraTr for *classification* (cross-entropy + softmax).
- In this benchmark, we adapt the head for *regression* by using a final Dense(1, linear).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="NIRBench")
def _gelu(x):
    # Keep an explicit GELU callable for layers that expect a function.
    return tf.nn.gelu(x)


@dataclass(frozen=True)
class SpectraTrConfig:
    """Convenience container (optional) for passing model hyperparameters."""
    input_dim: int
    n_channels: int = 1

    # Patchification
    patch_num: int = 40                  # Paper searches {5,10,20,25,40,50,80,100}
    pad_to_multiple: bool = True         # If input_dim not divisible by patch_num, pad with zeros.

    # Transformer
    embed_dim: int = 1024                # Not explicitly given in the paper; aligns with authors' ViT code default.
    depth: int = 5                       # Paper searches {2..9}
    num_heads: int = 12                  # Paper searches {8,10,12,14,16,18,20}
    dim_head: int = 64                   # Common ViT choice; authors' ViT code uses 64.
    mlp_dim: int = 512                   # Paper searches {128..2048}
    dropout: float = 0.0
    attn_dropout: float = 0.0

    # Positional embedding / pooling
    use_cls_token: bool = True
    pooling: str = "cls"                 # "cls" or "mean"
    pos_embedding: str = "gaussian_fixed"  # "gaussian_fixed" or "learned"
    pos_stddev: float = 0.02

    # Head
    head_units: Tuple[int, ...] = (128,)  # Paper does not specify MLP head depth/width.
    head_dropout: float = 0.0

    # Regularization
    l2_reg: float = 0.0                 # Paper tunes L2 in [0, 3e-2]


@tf.keras.utils.register_keras_serializable(package="NIRBench")
class ClsPool(layers.Layer):
    def call(self, t):
        return t[:, 0]


@tf.keras.utils.register_keras_serializable(package="NIRBench")
class _PatchEmbedding1D(layers.Layer):
    def __init__(
        self,
        input_dim: int,
        n_channels: int,
        patch_num: int,
        embed_dim: int,
        pad_to_multiple: bool,
        use_cls_token: bool,
        pos_embedding: str,
        pos_stddev: float,
        dropout: float,
        l2_reg: float,
        name: str = "patch_embed_1d",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if patch_num <= 0:
            raise ValueError("patch_num must be a positive integer.")
        self.input_dim = int(input_dim)
        self.n_channels = int(n_channels)
        self.patch_num = int(patch_num)
        self.embed_dim = int(embed_dim)
        self.pad_to_multiple = bool(pad_to_multiple)
        self.use_cls_token = bool(use_cls_token)
        self.pos_embedding = str(pos_embedding)
        self.pos_stddev = float(pos_stddev)
        self.dropout_rate = float(dropout)
        self.l2_reg = float(l2_reg)

        self.proj = layers.Dense(
            self.embed_dim,
            kernel_regularizer=keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
            name="linear_patch_projection",
        )
        self.drop = layers.Dropout(self.dropout_rate)

        # weights created in build() because sequence length depends on patch_num (+cls)
        self._cls_token = None
        self._pos_embed = None

    def build(self, input_shape):
        seq_len = self.patch_num + (1 if self.use_cls_token else 0)

        if self.use_cls_token:
            self._cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, self.embed_dim),
                initializer=keras.initializers.RandomNormal(stddev=self.pos_stddev),
                trainable=True,
            )

        trainable_pos = (self.pos_embedding.lower() == "learned")
        if self.pos_embedding.lower() not in {"gaussian_fixed", "learned"}:
            raise ValueError("pos_embedding must be 'gaussian_fixed' or 'learned'.")

        self._pos_embed = self.add_weight(
            name="pos_embedding",
            shape=(1, seq_len, self.embed_dim),
            initializer=keras.initializers.RandomNormal(stddev=self.pos_stddev),
            trainable=trainable_pos,
        )
        super().build(input_shape)

    def call(self, x, training: Optional[bool] = None):
        # Accept (B, L) or (B, L, C)
        if x.shape.rank == 2:
            x = tf.expand_dims(x, axis=-1)
        if x.shape.rank != 3:
            raise ValueError(f"Expected rank-2 or rank-3 input, got shape {x.shape}.")

        b = tf.shape(x)[0]
        l = tf.shape(x)[1]
        c = tf.shape(x)[2]

        if self.n_channels != 1:
            # If user specifies n_channels, enforce it (when statically known).
            if x.shape[-1] is not None and int(x.shape[-1]) != self.n_channels:
                raise ValueError(f"Expected n_channels={self.n_channels}, got {x.shape[-1]}.")

        # Compute patch_len = ceil(L / patch_num), pad if needed, then reshape
        patch_len = tf.cast(tf.math.ceil(tf.cast(l, tf.float32) / float(self.patch_num)), tf.int32)
        total_len = patch_len * self.patch_num

        if self.pad_to_multiple:
            pad_len = tf.maximum(0, total_len - l)
            x = tf.pad(x, paddings=[[0, 0], [0, pad_len], [0, 0]])
        else:
            # If not padding, require exact divisibility.
            tf.debugging.assert_equal(
                tf.math.floormod(l, self.patch_num),
                0,
                message="input length must be divisible by patch_num when pad_to_multiple=False",
            )
            patch_len = l // self.patch_num
            total_len = patch_len * self.patch_num
            x = x[:, :total_len, :]

        # (B, patch_num, patch_len, C) -> (B, patch_num, patch_len*C)
        x = tf.reshape(x, [b, self.patch_num, patch_len, c])
        x = tf.reshape(x, [b, self.patch_num, patch_len * c])

        # Linear patch projection
        x = self.proj(x)

        # Add CLS token if used
        if self.use_cls_token:
            cls = tf.tile(self._cls_token, [b, 1, 1])
            x = tf.concat([cls, x], axis=1)

        # Add positional embedding (fixed Gaussian by default, as described in the paper)
        x = x + self._pos_embed
        x = self.drop(x, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "input_dim": self.input_dim,
                "n_channels": self.n_channels,
                "patch_num": self.patch_num,
                "embed_dim": self.embed_dim,
                "pad_to_multiple": self.pad_to_multiple,
                "use_cls_token": self.use_cls_token,
                "pos_embedding": self.pos_embedding,
                "pos_stddev": self.pos_stddev,
                "dropout": self.dropout_rate,
                "l2_reg": self.l2_reg,
            }
        )
        return cfg


@tf.keras.utils.register_keras_serializable(package="NIRBench")
class _TransformerEncoderBlock(layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
        attn_dropout: float,
        l2_reg: float,
        name: str,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.dim_head = int(dim_head)
        self.mlp_dim = int(mlp_dim)
        self.dropout_rate = float(dropout)
        self.attn_dropout_rate = float(attn_dropout)
        self.l2_reg = float(l2_reg)

        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name="ln1")
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.dim_head,
            value_dim=self.dim_head,
            dropout=self.attn_dropout_rate,
            output_shape=self.embed_dim,
            name="mha",
        )
        self.drop1 = layers.Dropout(self.dropout_rate)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="ln2")
        self.ffn = keras.Sequential(
            [
                layers.Dense(
                    self.mlp_dim,
                    activation=_gelu,
                    kernel_regularizer=keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
                    name="fc1",
                ),
                layers.Dropout(self.dropout_rate),
                layers.Dense(
                    self.embed_dim,
                    kernel_regularizer=keras.regularizers.l2(self.l2_reg) if self.l2_reg > 0 else None,
                    name="fc2",
                ),
                layers.Dropout(self.dropout_rate),
            ],
            name="mlp",
        )

    def call(self, x, training: Optional[bool] = None):
        # Pre-norm attention + residual
        y = self.norm1(x)
        y = self.mha(y, y, training=training)
        y = self.drop1(y, training=training)
        x = x + y  # residual

        # Pre-norm MLP + residual
        y = self.norm2(x)
        y = self.ffn(y, training=training)
        x = x + y  # residual
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dim_head": self.dim_head,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout_rate,
                "attn_dropout": self.attn_dropout_rate,
                "l2_reg": self.l2_reg,
            }
        )
        return cfg


def build_spectratr(
    input_dim: int,
    *,
    n_channels: int = 1,
    patch_num: int = 40,
    pad_to_multiple: bool = True,
    embed_dim: int = 1024,
    depth: int = 5,
    num_heads: int = 12,
    dim_head: int = 64,
    mlp_dim: int = 512,
    dropout: float = 0.0,
    attn_dropout: float = 0.0,
    use_cls_token: bool = True,
    pooling: str = "cls",
    pos_embedding: str = "gaussian_fixed",
    pos_stddev: float = 0.02,
    head_units: Sequence[int] = (128,),
    head_dropout: float = 0.0,
    l2_reg: float = 0.0,
    name: str = "SpectraTr",
) -> keras.Model:
    """
    Build SpectraTr adapted for regression (Dense(1, linear) head).

    Parameters
    ----------
    input_dim:
        Number of spectral variables (wavelength points). First hyperparameter by benchmark convention.
    n_channels:
        Number of channels per spectral point (usually 1 for a single spectrum).
    patch_num:
        Number of patches to split the spectrum into. The paper reports optimal values such that
        each patch has ~50 points (e.g., 40 patches for 2074 points; 10 patches for 404 points).
    pad_to_multiple:
        If True, right-pad the spectrum with zeros so that it can be split into `patch_num` equal-length patches.
    embed_dim:
        Token embedding dimension (d_model). Not explicitly stated in the paper; exposed as a hyperparameter.
    depth:
        Number of stacked transformer encoder blocks.
    num_heads:
        Number of attention heads.
    dim_head:
        Dimension per head used by Keras MultiHeadAttention.
    mlp_dim:
        Hidden dimension of the per-token feed-forward MLP inside each transformer block.
    dropout / attn_dropout:
        Dropout probabilities.
    use_cls_token:
        Whether to prepend a CLS token and pool from it.
    pooling:
        "cls" (use CLS token output) or "mean" (global average over tokens).
    pos_embedding:
        "gaussian_fixed" (non-trainable Gaussian positional embeddings) or "learned" (trainable).
    head_units:
        Hidden units in the regression head MLP after pooling.
    head_dropout:
        Dropout in the regression head.
    l2_reg:
        L2 regularization coefficient applied to Dense kernels (projection + MLPs + head).
    """
    if pooling not in {"cls", "mean"}:
        raise ValueError("pooling must be 'cls' or 'mean'.")
    if pooling == "cls" and not use_cls_token:
        raise ValueError("pooling='cls' requires use_cls_token=True.")

    inputs = keras.Input(shape=(int(input_dim),) if n_channels == 1 else (int(input_dim), int(n_channels)), name="x")

    x = _PatchEmbedding1D(
        input_dim=int(input_dim),
        n_channels=int(n_channels),
        patch_num=int(patch_num),
        embed_dim=int(embed_dim),
        pad_to_multiple=bool(pad_to_multiple),
        use_cls_token=bool(use_cls_token),
        pos_embedding=str(pos_embedding),
        pos_stddev=float(pos_stddev),
        dropout=float(dropout),
        l2_reg=float(l2_reg),
        name="patch_embedding",
    )(inputs)

    for i in range(int(depth)):
        x = _TransformerEncoderBlock(
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            dim_head=int(dim_head),
            mlp_dim=int(mlp_dim),
            dropout=float(dropout),
            attn_dropout=float(attn_dropout),
            l2_reg=float(l2_reg),
            name=f"encoder_block_{i+1}",
        )(x)

    x = layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)

    if pooling == "cls":
        x = ClsPool(name="cls_pool")(x)
    else:
        x = layers.GlobalAveragePooling1D(name="mean_pool")(x)

    # Head MLP (not specified in detail in the paper; exposed as hyperparameters)
    for j, units in enumerate(tuple(head_units)):
        x = layers.Dense(
            int(units),
            activation=_gelu,
            kernel_regularizer=keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
            name=f"head_fc{j+1}",
        )(x)
        if head_dropout and head_dropout > 0:
            x = layers.Dropout(float(head_dropout), name=f"head_dropout{j+1}")(x)

    outputs = layers.Dense(
        1,
        activation="linear",
        kernel_regularizer=keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
        name="y",
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name=name)


# -----------------------------------------------------------------------------
# Paper-reported hyperparameters (classification setting; adapted here to regression)
# -----------------------------------------------------------------------------
#
# From the paper (SpectraTr, J. Innov. Opt. Health Sci., 2022):
# - Task: qualitative analysis (classification).
# - Loss: cross-entropy; Optimizer: Adam; epochs: 200. Learning-rate halving if
#   training loss does not decline within 10 epochs; early stopping if test loss
#   does not drop within 30 epochs. (See Sec. 2.3)
#
# Hyperparameter search ranges (Table 1):
# - Batch size: {16, 32, 64, 128}
# - Learning rate: 1e-4 .. 1e-1
# - L2 regularization: 0 .. 3e-2
# - Patch num: {5, 10, 20, 25, 40, 50, 80, 100}
# - Multi-head num: {8, 10, 12, 14, 16, 18, 20}
# - Layer depth: {2, 3, 4, 5, 6, 7, 8, 9}
# - MLP dim: {128, 256, 512, 1024, 1536, 2048}
#
# Final selected parameters (Table 2; two datasets):
# - Dataset A (2074 points): batch=16, lr=1e-4, L2=0.01, patch_num=40, heads=19, depth=5, mlp_dim=512
# - Dataset B (404 points):  batch=16, lr=1e-4, L2=0.00, patch_num=10, heads=12, depth=3, mlp_dim=1024
#
# Missing/unclear implementation details in the paper:
# - Token embedding dimension (d_model / embed_dim) is not explicitly stated.
# - Whether positional embeddings are fixed or trainable is not explicit; the paper
#   describes Gaussian position embeddings, so this implementation defaults to a
#   non-trainable Gaussian positional embedding but allows 'learned'.
# - Activation functions and head MLP depth/width are not fully specified; we use
#   GELU (common in ViT-style models) and expose head_units as a hyperparameter.
