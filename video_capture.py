import cv2 as opencv

# Camera capture
camera_capture = opencv.VideoCapture(0)

ms_delay = 25

is_template_set = False

def draw_rectangle(camera_frame, template, max_value, max_location):
    w, h, _ = template.shape

    top_left_corner = (max_location[0], max_location[1])
    bottom_right_corner = (max_location[0] + h, max_location[1] + w)

    corr_threshold = 0.35

    if(max_value > corr_threshold):
        color = (0, 255, 0) # BGR
    else:
        color = (0, 0, 255) # BGR

    opencv.rectangle(camera_frame, top_left_corner, bottom_right_corner, color, 3)

ignore_first_frame = True

while(True):
    capture_status, camera_frame = camera_capture.read()    

    if(ignore_first_frame):
        ignore_first_frame = False
        continue

    if(capture_status):
        opencv.imshow('Camera preview', camera_frame)
        
    if(is_template_set == False):
        offset = 125
        template = camera_frame[offset:-offset, offset:-offset]
        opencv.imshow('Template', template)

        is_template_set = True

    match_result = opencv.matchTemplate(camera_frame, template, opencv.TM_CCOEFF_NORMED)
    opencv.imshow('Match result', opencv.pow(match_result, 3.0))

    min_value, max_value, min_loc, max_loc = opencv.minMaxLoc(match_result)
    print('Min: {0:.2f}, Max: {1:.2f}'.format(min_value, max_value))

    draw_rectangle(camera_frame, template, max_value, max_loc)
    opencv.imshow('Camera preview with rectangle', camera_frame)

    key_pressed = opencv.waitKey(ms_delay)
    if(key_pressed == ord('q')):
        break
