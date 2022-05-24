import cv2 as opencv
import numpy as np
import math

feature_detector = opencv.AKAZE_create()
feature_matcher = opencv.DescriptorMatcher_create(
    opencv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

def translate_rotate_and_scale_image(image, T, R, S):
    rows, cols, _ = image.shape

    transformation_center = (cols / 2, rows / 2)

    transformation_matrix = opencv.getRotationMatrix2D(transformation_center, R, S)

    # Add translation
    transformation_matrix[0, 2] += T[0]
    transformation_matrix[1, 2] += T[1]

    return opencv.warpAffine(image, transformation_matrix, None)

def find_features(input_image, display_image, window_name=''):
    input_image_features, input_image_descriptor = feature_detector.detectAndCompute(input_image, None)

    image_with_keypoints = opencv.drawKeypoints(input_image, input_image_features, None,
        flags = opencv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS | opencv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if(display_image):
        opencv.imshow(window_name, image_with_keypoints)

    return input_image_features, input_image_descriptor

def filter_matches(matches, top_matches_count=15):
    matches.sort(key = lambda x: x.distance, reverse = False)
    return matches[:top_matches_count]

def find_homography(template_key_pts, image_key_pts, matches):

    # Get key points of matched descriptors
    template_pts = [template_key_pts[match.queryIdx].pt for match in matches]
    current_frame_pts = [image_key_pts[match.trainIdx].pt for match in matches]

    # Find and then return homography
    H = opencv.findHomography(np.array(current_frame_pts), np.array(template_pts))
    return H[0]

def perform_feature_matching(template_image, input_image):
    
    # Compute features
    template_keypoints, template_desc = find_features(template_image, False)
    test_keypoints, test_desc = find_features(input_image, False)

    # Match features
    matches = feature_matcher.match(template_desc, test_desc, None)

    # Filter matches
    matches = filter_matches(matches)
                
    # Find and return homography
    return find_homography(template_keypoints, test_keypoints, matches)


def draw_tracking_result(H, template, test_image):
    # Get template dimensions
    height, width, _ = template.shape
        
    # Prepare template rectangle
    template_rect = np.float32([[0, 0], [0, height], 
                                [width, height], [width, 0]]);

    # Transform rectangle using homography        
    template_rect_transformed = opencv.perspectiveTransform(
        template_rect.reshape(-1,1,2), np.linalg.inv(H))        

    # Draw rectangle
    opencv.polylines(test_image, [np.int32(template_rect_transformed)], True, 
                    (0,255,0), 3)     
            
    # Return frame with rectangle
    return test_image