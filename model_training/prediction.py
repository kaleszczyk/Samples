from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

# Model loading
model = keras.models.load_model('model-thrash')

# Get images
IMG_HEIGHT = 256
IMG_WIDTH = 192
input_dir = 'c:\\Users\\Dawid\\MLTraining\\datasets\\Thrash\\Validation\\'

test_images = image_dataset_from_directory(input_dir, 
                                            image_size=(IMG_HEIGHT, IMG_WIDTH),
                                            color_mode='grayscale')

# Prediction
class_names = ['Cardboard', 'Glass']

plt.figure(figsize=(10, 10))
image_count_to_test = 10
for images, labels in test_images.take(1):
    print(images.shape)
    prediction_result = model.predict(images)

    for i in range(image_count_to_test):
        ax = plt.subplot(5, 2, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
        
        if(prediction_result[i] > 0.5):
            predicted_label_index = 1
        else:
            predicted_label_index = 0

        plt.title(class_names[int(labels[i])] + '(' + class_names[predicted_label_index] + ')')
                
        plt.axis("off")

plt.show()

