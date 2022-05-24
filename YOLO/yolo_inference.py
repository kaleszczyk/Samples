import cv2 as opencv
import numpy as np

class YoloInference(object):
    def __init__(self, config_file_path, weights_file_path, labels_file_path):
        # Load model
        self.load_model_and_configure(config_file_path, weights_file_path)

        # Load labels
        self.load_labels_from_file(labels_file_path)
    
    def load_model_and_configure(self, config_file_path, weights_file_path):        
        # Load YOLO
        self.interpreter = opencv.dnn.readNetFromDarknet(config_file_path, weights_file_path)

        # Get output layers 
        layer_names = self.interpreter.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.interpreter.getUnconnectedOutLayers()]

        # Set the input image size accepted by YOLO and scaling factor
        self.input_image_size = (608, 608)
        self.scaling_factor = 1 / 255.0
        
    def load_labels_from_file(self, file_path):        
        with open(file_path, 'r') as file:
            self.labels = [line.strip() for line in file.readlines()]

    def prepare_image(self, image):    
        # Converts image to the blob using scaling factor and input image size accepted by YOLO
        blob = opencv.dnn.blobFromImage(image, self.scaling_factor, 
            self.input_image_size, swapRB=True, crop=False)

        return blob

    def convert_bounding_box_to_rectangle_points(self, bounding_box, input_image_size):
        # Get image width and height
        width = input_image_size[0]
        height = input_image_size[1]            
        
        # Get box width and height
        box_width = bounding_box[2]
        box_height = bounding_box[3]

        # YOLO returns box center. So, we convert the first two elements 
        # to the coordinates of the top left corner
        x_box = bounding_box[0] - box_width / 2.0
        y_box = bounding_box[1] - box_height / 2.0

        # Get the top-left and bottom-right corner of the rectangle 
        # surrounding the object in the image
        top_left_corner = (int(x_box * width), int(y_box* height))
        bottom_right_corner = (int((x_box + box_width) * width), 
            int((y_box + box_height) * height))                

        return (top_left_corner, bottom_right_corner)

    def adjust_bounding_box_to_image(self, bounding_box, input_image_size):
        # Get image width and height
        width = input_image_size[0]
        height = input_image_size[1]            
        
        box_width = bounding_box[2]
        box_height = bounding_box[3]

        # YOLO returns box center. So, we convert the first two elements 
        # to the coordinates of the top left corner
        x_box = bounding_box[0] - box_width / 2.0
        y_box = bounding_box[1] - box_height / 2.0

        return [int(x_box * width), 
            int(y_box * height), 
            int((x_box + box_width) * width),
            int((y_box + box_height) * height)]

    def get_object_label_and_detection_score(self, detection_result):
        scores = detection_result[5:]    

        class_id = np.argmax(scores)

        return self.labels[class_id], scores[class_id]

    def parse_detection_result(self, input_image_size, detection_result, threshold):        
        # Get the object label and detection score
        label, score = self.get_object_label_and_detection_score(detection_result)
        
        # Store only objects with the score above the threshold and label 'person'
        if(score > threshold and label == 'person'):
            box = detection_result[0:4]
            
            return {
                    'rectangle': self.convert_bounding_box_to_rectangle_points(box, input_image_size),
                    'label': label,
                    'score': float(score),
                    'box' : self.adjust_bounding_box_to_image(box, input_image_size)
                }
        else:
            return None

    def get_values_from_detection_results_by_key(self, detection_results, dict_key):        
        return [detection_results[i][dict_key] for i in range(0, len(detection_results))]

    def filter_detections(self, detected_people, threshold, nms_threshold):
        # Get scores and boxes
        scores = self.get_values_from_detection_results_by_key(detected_people, 'score')
        boxes = self.get_values_from_detection_results_by_key(detected_people, 'box')
        
        # Get best detections
        best_detections_indices = opencv.dnn.NMSBoxes(boxes, scores, threshold, nms_threshold)                

        # Return filtered people
        return [detected_people[i] for i in best_detections_indices.flatten()]

    def detect_people(self, image, threshold):
        # Store the original image size
        input_image_size = image.shape[-2::-1]

        # Preprocess image to get the blob
        image = self.prepare_image(image)

        # Set the blob as the interpreter (neural network) input
        self.interpreter.setInput(image)

        # Run inference
        output_layers = self.interpreter.forward(self.output_layers)
        
        # Process output layers
        detected_people = []        
        for output_layer in output_layers:            
            for detection_result in output_layer:                
                object_info = self.parse_detection_result(input_image_size, 
                    detection_result, threshold)                

                if(object_info is not None):                    
                    detected_people.append(object_info)

        # Filter out overlapping detections
        nms_threshold = 0.75
        detected_people = self.filter_detections(detected_people, threshold, nms_threshold)
        
        return detected_people