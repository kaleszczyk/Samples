import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D, experimental

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

# Base folder for dataset
base_dir = 'c:\\Users\\Dawid\\MLTraining\\datasets\\Thrash\\'

train_dir = os.path.join(base_dir, 'Training')
validation_dir = os.path.join(base_dir, 'Validation')

# Training parameters
epochs = 10
IMG_HEIGHT = 256
IMG_WIDTH = 192
batch_size = 32

# Generators
train_image_generator = ImageDataGenerator()

validation_image_generator = ImageDataGenerator()

# Training images flow
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            batch_size=batch_size,
                                                            color_mode='grayscale',
                                                            class_mode='binary')

val_data_gen = train_image_generator.flow_from_directory(directory=validation_dir,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            batch_size=batch_size,
                                                            color_mode='grayscale',
                                                            class_mode='binary')

# Simple preview
sample_training_images, _ = next(train_data_gen)

plt.imshow(sample_training_images[-1], cmap='gray')
plt.show()

# Model 
model  = Sequential([
    experimental.preprocessing.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),

    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),

    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),

    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
model.summary()

def get_number_of_images_in_dir(root_dir, class_name):   
    dir = os.path.join(root_dir, class_name)
    return len(os.listdir(dir))    

num_cardboard_tr = get_number_of_images_in_dir(train_dir, 'Cardboard')
num_glass_tr = get_number_of_images_in_dir(train_dir, 'Glass')

num_cardboard_val = get_number_of_images_in_dir(validation_dir, 'Cardboard')
num_glass_val = get_number_of_images_in_dir(validation_dir, 'Glass')

#print(num_cardboard_tr)

# Training
train_cardboard_dir = os.path.join(train_dir, 'Cardboard')
num_cardboard_tr = len(os.listdir(train_cardboard_dir))
#print(num_cardboard_tr)

train_glass_dir = os.path.join(train_dir, 'Glass')
num_glass_tr = len(os.listdir(train_glass_dir))

validation_cardboard_dir = os.path.join(validation_dir, 'Cardboard')
num_cardboard_val = len(os.listdir(validation_cardboard_dir))

validation_glass_dir = os.path.join(validation_dir, 'Glass')
num_glass_val = len(os.listdir(validation_glass_dir))

total_train = num_cardboard_tr + num_glass_tr
total_val = num_cardboard_val + num_glass_val

history = model.fit(train_data_gen,
                    steps_per_epoch=total_train // batch_size,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=total_val // batch_size)

# Model saving
model.save('model-thrash', save_format='tf')

# Model efficiency
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.ylim([0, 1.1])
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.ylim([0, 1.1])
plt.legend(loc='best')

plt.show()