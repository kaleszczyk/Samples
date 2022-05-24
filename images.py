import cv2 as opencv

# Wczytywanie obrazu
lena_img = opencv.imread('Images\Lena.png', opencv.IMREAD_GRAYSCALE)

# Prezentacja
opencv.imshow('Lena image', lena_img)
opencv.waitKey()

# Process image
kernel_size = (11, 11)
sigma_X = 0
lena_img_processed = opencv.GaussianBlur(lena_img, kernel_size, sigma_X)

# Display processed image
opencv.imshow('Lena blurred', lena_img_processed)
opencv.waitKey()

# Zapisywanie
opencv.imwrite('Images\Lena-processed.png', lena_img_processed)