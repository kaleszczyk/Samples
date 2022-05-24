from turtle import xcor
from PIL import Image
import numpy as np

#load image
image = Image.open(r"C:/Users/Kasia/Desktop/lena.png")

#convert image to array 
image_arr = np.array(image)
print(image_arr.shape[1])
print(image_arr.shape[0])
#crop image
def crop_image(image_to_crop, x, y):
    random_strideX = np.random.randint(0, image_to_crop.shape[0]-x)
    random_strideY = np.random.randint(0, image_to_crop.shape[1]-y)
    if x >= 0 and x < image_to_crop.shape[0] and y >= 0 and y<image_to_crop.shape[1]: 
        image_result = image_to_crop[x:random_strideX, y:random_strideY]
        return Image.fromarray(image_result)
    else: 
        return Image.fromarray(image_to_crop)
             
i=0
while i<10: 
    img_cropped = crop_image(image_arr, 0, 0)
    img_cropped.show()
    i = i +1

