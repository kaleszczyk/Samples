import cv2 as opencv

input_window_name = 'Lena'
output_window_name = 'Lena processed'

initial_blur = 3

def process_image(trackbar_value):
    if(trackbar_value % 2 == 1):
        lena_image_processed = opencv.medianBlur(lena_image, trackbar_value)
        opencv.imshow(output_window_name, lena_image_processed)

# Tworzenie okien
opencv.namedWindow(input_window_name, opencv.WINDOW_AUTOSIZE)
opencv.namedWindow(output_window_name, opencv.WINDOW_AUTOSIZE)

# Tworzenie suwaka
opencv.createTrackbar('Blur size', input_window_name, initial_blur, 51, process_image)

# Wyswietlanie obrazu
lena_image = opencv.imread('Images\Lena.png')
opencv.imshow(input_window_name, lena_image)

opencv.waitKey()