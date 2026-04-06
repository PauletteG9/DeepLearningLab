import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_generator(latent_dim = 100):
  model = tf.keras.Sequential([
    layers.Dense(7*7*128, input_dim=latent_dim),
    layers.Reshape((7, 7, 128)),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),

    layers.Conv2DTranspose(64, kernel_size=3, strides=2, paddding='same'),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),

    layers.Conv2DTranspose(32, kernel_size=3, strides=2, paddding='same'),
    layers.BatchNormalization()
    layers.LeakyReLU(0.2)

    layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding='same'. activation='tanh'),
  ], name="Generator")
  return model
