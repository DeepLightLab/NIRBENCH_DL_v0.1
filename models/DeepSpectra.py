"""
DeepSpectra Model Implementation

This module contains the implementation of the DeepSpectra CNN model for spectral analysis,
based on the architecture described in the paper "DeepSpectra: An end-to-end deep learning 
approach for quantitative spectral analysis" (Analytica Chimica Acta 1058 (2019) 48-57).
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LeakyReLU, MaxPooling1D, concatenate, Flatten, BatchNormalization, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

__all__ = ['deepspectra_model']

def deepspectra_model(input_vector_dimension, 
                        kernel_size_1, 
                        kernel_size_2, 
                        kernel_size_3,
                        stride_1, 
                        stride_2, 
                        hidden_number, 
                        dropout_rate,
                        l2_regularization):
    """
    Builds the DeepSpectra CNN model as described in the paper.
    Analytica Chimica Acta 1058 (2019) 48-57. 
    Implementation by: D.Passos (dmpassos@uagl.pt)
    Args:
        input_vector_dimension (int): The number of features in the input spectra.
        kernel_size_1 (int): Kernel size for the first convolutional layer (Conv1).
        kernel_size_2 (int): Kernel size for the second main branch of the Inception module.
        kernel_size_3 (int): Kernel size for the third main branch of the Inception module.
        stride_1 (int): Stride for the first convolutional layer (Conv1).
        stride_2 (int): Stride for the parallel branches in the Inception module.
        hidden_number (int): Number of neurons in the fully connected layer (F1).
        dropout_rate (float): The dropout rate to be applied.
        l2_regularization (float): The L2 regularization factor.

    Returns:
        tensorflow.keras.Model: The compiled Keras DeepSpectra model.
    """
    
    # Define common layer parameters
    kernel_initializer = tf.keras.initializers.HeNormal(seed=12345)
    kernel_regularizer = l2(l2_regularization)
    
    # Input layer expects 3D tensor of shape (batch_size, steps, channels)
    # and channels=1.
    input_layer = Input(shape=(input_vector_dimension, 1))

    # Layer Conv1
    # Paper specifies 8 filters for this layer.
    conv1 = Conv1D(filters=8,
                   kernel_size=kernel_size_1,
                   strides=stride_1,
                   padding='same',
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer)(input_layer)
    conv1 = LeakyReLU()(conv1)

    # --- Inception Module ---
    # According to Figure 1, each of the 4 branches outputs 4 filters.
    # We will also use 4 filters for the reduction layers.
    num_inception_filters = 4

    # Branch 1: 1x1 Convolution
    tower_1 = Conv1D(filters=num_inception_filters,
                     kernel_size=1,
                     strides=stride_2,
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer)(conv1)
    tower_1 = LeakyReLU()(tower_1)

    # Branch 2: 1x1 Conv -> 3x1 Conv (or kernel_size_2)
    tower_2 = Conv1D(filters=num_inception_filters,
                     kernel_size=1,
                     strides=1, # Reduction layer has stride 1
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer)(conv1)
    tower_2 = LeakyReLU()(tower_2)
    tower_2 = Conv1D(filters=num_inception_filters,
                     kernel_size=kernel_size_2,
                     strides=stride_2,
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer)(tower_2)
    tower_2 = LeakyReLU()(tower_2)

    # Branch 3: 1x1 Conv -> 5x1 Conv (or kernel_size_3)
    tower_3 = Conv1D(filters=num_inception_filters,
                     kernel_size=1,
                     strides=1, # Reduction layer has stride 1
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer)(conv1)
    tower_3 = LeakyReLU()(tower_3)
    tower_3 = Conv1D(filters=num_inception_filters,
                     kernel_size=kernel_size_3,
                     strides=stride_2,
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer)(tower_3)
    tower_3 = LeakyReLU()(tower_3)

    # Branch 4: Max Pooling -> 1x1 Conv
    tower_4 = MaxPooling1D(pool_size=3,
                           strides=stride_2,
                           padding='same')(conv1)
    tower_4 = Conv1D(filters=num_inception_filters,
                     kernel_size=1,
                     strides=1, # Conv after pool has stride 1
                     padding='same',
                     kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer)(tower_4)
    tower_4 = LeakyReLU()(tower_4)

    # Concatenate the outputs of the Inception module branches
    inception_output = concatenate([tower_1, tower_2, tower_3, tower_4], axis=-1)

    # --- Fully Connected Stage ---
    
    # Flatten layer
    flatten_layer = Flatten()(inception_output)

    # Batch Normalization and Dropout after Flatten
    bn1 = BatchNormalization()(flatten_layer)
    dropout1 = Dropout(dropout_rate)(bn1)
   
    
    # F1 Layer (Dense)
    f1 = Dense(units=hidden_number,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=kernel_regularizer)(dropout1)
    f1 = LeakyReLU()(f1)

    # Batch Normalization and Dropout after F1
    bn2 = BatchNormalization()(f1)
    dropout2 = Dropout(dropout_rate)(bn2)

    # Output Layer (1 neuron for regression, linear activation)
    output_layer = Dense(units=1)(dropout2)
    output_layer = LeakyReLU()(output_layer) # Using LeakyReLU as per the paper, but could also use linear activation!

    # Create and return the model
    model = Model(inputs=input_layer, outputs=output_layer, name='DeepSpectra')
    
    return model