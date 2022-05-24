import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2


from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from glob import glob


input_dirs_train = glob('D:\\Autoencoder\\dataset_wieksza_czesc_poszerzony_zbior\\train\\'+ '/*.png')
input_dirs_test = glob('D:\\Autoencoder\\dataset_wieksza_czesc_poszerzony_zbior\\test\\'+ '/*.png')
input_dirs_val = glob('D:\\Autoencoder\\dataset_wieksza_czesc_poszerzony_zbior\\val\\'+ '/*.png')

images_train = []
images_test = []
images_val = []
width = 200
height = 200
dsize = (width, height) 
for input_dir in input_dirs_train:
    image_file = cv2.imread(input_dir, 0)
    image_file = cv2.resize(image_file, dsize)
    image_file = cv2.Sobel(image_file, cv2 
    image_file = image_file.astype('float32') / 255
    images_train.append(image_file)
    #image = cv2.resize(image_file, size)

for input_dir in input_dirs_test:
    image_file = cv2.imread(input_dir, 0)
    image_file = cv2.resize(image_file, dsize)
    image_file = cv2.Sobel(image_file)
    image_file = image_file.astype('float32') / 255
    images_test.append(image_file)

for input_dir in input_dirs_val:
    image_file = cv2.imread(input_dir, 0)
    image_file = cv2.resize(image_file, dsize)
    image_file = image_file.astype('float32') / 255
    images_val.append(image_file)      

x_train = np.array(images_train)
x_test = np.array(images_test)
x_val = np.array(images_val)

#(x_train, _), (x_test, _) = fashion_mnist.load_data()

#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.

print (x_train.shape)
print (x_test.shape)
print (x_val.shape)

latent_dim = 64  #64-wymiarowy utajony wektor

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(40000, activation='sigmoid'),
      layers.Reshape((200, 200))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train, x_train,
                epochs=20,
                shuffle=True,
                validation_data=(x_test, x_test))

#walidacja
encoded_imgs = autoencoder.encoder(x_val).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

#reconstructions = autoencoder.predict(x_train)
#train_loss = tf.keras.losses.mae(reconstructions, x_train)

#plt.hist(train_loss, bins=50)
#plt.xlabel("Train loss")
#plt.ylabel("No of examples")
#plt.show()

#reconstructions = autoencoder.predict(x_val)
#val_loss = tf.keras.losses.mae(reconstructions, x_val)

#plt.hist(val_loss, bins=50)
#plt.xlabel("Val loss")
#plt.ylabel("No of examples")
#plt.show()

n = 5
plt.figure(figsize=(200, 200))
for i in range(n):
  #calculate a reconstruction cost 
  cost = decoded_imgs[i] - x_val[i]
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_val[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

plt.show()
