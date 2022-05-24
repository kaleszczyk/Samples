import helpers
import cv2 as opencv

input_image = opencv.imread('Images\\lena.png')

offset =160
template_image = input_image[offset:-offset, offset:-offset]

input_image = helpers.translate_rotate_and_scale_image(input_image, [10, 0], 10, 1)

H = helpers.perform_feature_matching(template_image, input_image)
print(H)

input_image = helpers.draw_tracking_result(H, template_image, input_image)

opencv.imshow('Template', template_image)
opencv.imshow('Test', input_image)
opencv.waitKey(0)