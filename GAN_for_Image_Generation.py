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

def build_discriminator(img_shape=(28, 28, 1)):
  model = tf.keras.Sequential([
    layers.Conv2D(32, kernel_size=3, strides=2, padding='same', input_shape=img_shape),
    layers.LeakyReLU(0.2),
    layers.Dropout(0.3),                
    
    layers.Conv2D(64, kernel_size=3, strides=2, padding='same'),
    layers.LeakyReLU(0.2),
    layers.Dropout(0.3),
    
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid'),   # VIVA: sigmoid → probability; 1=real, 0=fake
  ], name="Discriminator")
  return model

def train_gan(epochs=5, batch_size=64, latent_dim=100):
  (X_train, _), _ = tf.keras.datasets.mnist.load_data()
  X_train = (X_train.astype('float32') - 127.5) / 127.5  
  X_train = np.expand_dims(X_train, axis=-1)          
  
  generator     = build_generator(latent_dim)
  discriminator = build_discriminator()
  
  discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
  
  discriminator.trainable = False
  gan_input  = layers.Input(shape=(latent_dim,))
  fake_img   = generator(gan_input)
  validity   = discriminator(fake_img)
  gan        = Model(gan_input, validity, name="GAN")
  gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),loss='binary_crossentropy')

  real_labels = np.ones((batch_size, 1))    # VIVA: real images labeled as 1
  fake_labels = np.zeros((batch_size, 1))   # VIVA: generated images labeled as 0

  for epoch in range(epochs):
    idx       = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]
    noise     = np.random.normal(0, 1, (batch_size, latent_dim))  # Sample from latent space
    fake_imgs = generator.predict(noise, verbose=0)
    
    d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
    d_loss      = 0.5 * np.add(d_loss_real, d_loss_fake)  # VIVA: average D loss over real+fake
    
    noise  = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, real_labels)
    
    print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss[0]:.4f} | G Loss: {g_loss:.4f}")
    
  noise     = np.random.normal(0, 1, (16, latent_dim))
  gen_imgs  = generator.predict(noise, verbose=0)
  gen_imgs  = 0.5 * gen_imgs + 0.5       
  
  fig, axes = plt.subplots(4, 4, figsize=(6, 6))
  for i, ax in enumerate(axes.flat):
    ax.imshow(gen_imgs[i, :, :, 0], cmap='gray')
    ax.axis('off')
  plt.suptitle("GAN Generated Images")
  plt.tight_layout()
  plt.savefig("gan_generated.png", dpi=100)
  plt.show()
  print("Saved: gan_generated.png")

  generator.summary()
  discriminator.summary()

train_gan(epochs=5)
