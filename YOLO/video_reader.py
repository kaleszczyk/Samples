import cv2 as opencv

class VideoReader(object):
    def __init__(self, file_path):
        # Open the video file        
        try:
            self.video_capture = opencv.VideoCapture(file_path)
        except expression as identifier:
            print(identifier)
                
    def read_next_frame(self):                    
        (capture_status, frame) = self.video_capture.read()
        
        # Verify the status
        if(capture_status):
            return frame

        else:
            return None