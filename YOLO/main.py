from yolo_inference import YoloInference as model
from image_helper import ImageHelper as imgHelper
from video_reader import VideoReader as videoReader

# Load and prepare model
config_file_path = 'Models/03_yolo.cfg'    
weights_file_path = 'Models/04_yolo.weights'
labels_file_path = 'Models/05_yolo-labels.txt'

# Initialize model
ai_model = model(config_file_path, weights_file_path, labels_file_path)

# Initialize video reader
video_file_path = 'Videos/01.mp4'
video_reader = videoReader(video_file_path)

# Detection and preview parameters
score_threshold = 0.5
delay_between_frames = 5

# Perform object detection in the video sequence
while(True):
    # Get frame from the video file
    frame = video_reader.read_next_frame()

    # If frame is None, then break the loop
    if(frame is None):
        break

    # Perform detection        
    results = ai_model.detect_people(frame, score_threshold)

    imgHelper.display_image_with_detected_objects(frame, results, delay_between_frames)    