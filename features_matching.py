import cv2 as opencv
import numpy as np

feature_detector = opencv.AKAZE_create()

input_image = opencv.imread(r"C:/Users/Kasia/Desktop/lena.png")


def translate_rotate_and_scale_image(image, T, R, S):
    rows, cols, _ = image.shape

    transformation_center = (cols / 2, rows / 2)

    transformation_matrix = opencv.getRotationMatrix2D(transformation_center, R, S)

    # Add translation
    transformation_matrix[0, 2] += T[0]
    transformation_matrix[1, 2] += T[1]

    return opencv.warpAffine(image, transformation_matrix, None)

def find_features(input_image, display_image, window_name):
    input_image_features, input_image_descriptor = feature_detector.detectAndCompute(input_image, None)

    image_with_keypoints = opencv.drawKeypoints(input_image, input_image_features, None,
        flags = opencv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS | opencv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if(display_image):
        opencv.imshow(window_name, image_with_keypoints)

    return input_image_features, input_image_descriptor

def filter_matches(matches, top_matches_count=15):
    matches.sort(key = lambda x: x.distance, reverse = False)

    return matches[:top_matches_count]

T =[50, 50]
R = -25
S = 1.5
input_image_transformed = translate_rotate_and_scale_image(input_image, T, R, S)

#opencv.imshow('Input image', input_image)
#opencv.imshow('Input image transformed', input_image_transformed)

input_image_features, input_image_descriptor = find_features(
    input_image, True, 'Input with features')

trans_image_features, trans_image_descriptor = find_features(
    input_image_transformed, True, 'Input with features')

# Feature matching
feature_matcher = opencv.DescriptorMatcher_create(
    opencv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

matches = feature_matcher.match(input_image_descriptor, trans_image_descriptor, None)

# Display matches
matches = filter_matches(matches)
image_with_matches = opencv.drawMatches(input_image, input_image_features,
    input_image_transformed, trans_image_features, matches, None)

# Get points
template_pts = [input_image_features[match.queryIdx].pt for match in matches]
test_pts = [trans_image_features[match.trainIdx].pt for match in matches]

# Find transformation matrix
#H = opencv.findHomography(np.array(template_pts), np.array(test_pts))
H = opencv.findHomography(np.array(test_pts),np.array(template_pts))
H = H[0]

print(H)

opencv.imshow('Matched features', image_with_matches)

# Registration
registered_image = opencv.warpPerspective(input_image_transformed, H, None)
opencv.imshow('Registered', registered_image)

opencv.waitKey(0)