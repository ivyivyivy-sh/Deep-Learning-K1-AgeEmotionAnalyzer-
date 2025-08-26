import tensorflow as tf
import keras
import cv2
import numpy as np
import pandas as pd
import skimage as ski
import av
import os
from streamlit.runtime.uploaded_file_manager import UploadedFile

keras.config.disable_interactive_logging()

# Emotion classification labels
emotion_labels = list(map(str.lower, ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']))
# Age range labels (matching the pre-trained model output)
age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

class Analyzer:
    """ Audience Emotion Analyzer class with age prediction capability """
    
    def __init__(self) -> None:
        # Initialize face detector using Haar cascade
        self.detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        
        # Create results DataFrame with columns for frame, emotions, ages, and face coordinates
        columns = ["frame"] + emotion_labels + age_labels + ["x", "y", "width", "height"]
        self.results = pd.DataFrame(columns=columns)
        
        # Initialize age prediction model
        self.age_net = None
        self.initialize_age_model()

    def initialize_age_model(self):
        """Initialize the age prediction model using OpenCV DNN"""
        try:
            # Model architecture file (prototxt)
            age_proto = "C:\\Users\\guest_ivy\\Desktop\\kunskapskontroll_1_new\\deploy.prototxt"
            # Model weights file (using your specified path)
            age_weights = "C:\\Users\\guest_ivy\\Desktop\\kunskapskontroll_1_new\\age_net.caffemodel"
            
            # Verify that model files exist
            if not os.path.exists(age_proto):
                print(f"Age protocol file not found: {age_proto}")
                return
            if not os.path.exists(age_weights):
                print(f"Age weights file not found: {age_weights}")
                return
            
            # Load the age prediction model using OpenCV's DNN module
            self.age_net = cv2.dnn.readNetFromCaffe(age_proto, age_weights)
            print("Age prediction model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load age prediction model: {e}")
            self.age_net = None

    def predict_age(self, face_image):
        """Predict age distribution for a given face image using OpenCV DNN"""
        # Return zeros if model is not loaded
        if self.age_net is None:
            return np.zeros(len(age_labels))
        
        try:
            # Preprocess image for the age prediction model
            # - Rescale to 1.0, resize to 227x227 (model input size)
            # - Apply mean subtraction values (BGR channels)
            # - swapRB=False because OpenCV uses BGR format
            blob = cv2.dnn.blobFromImage(
                face_image, 
                1.0, 
                (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False
            )
            
            # Set input and perform forward pass through the network
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            
            # Return the age prediction probabilities
            return age_preds[0]
            
        except Exception as e:
            print(f"Error during age prediction: {e}")
            return np.zeros(len(age_labels))

    def analyze(self, 
        model = None,
        file: UploadedFile | None = None, 
        skip: int | None = 1,
        confidence: float | None = .5) -> tuple[bool, pd.DataFrame]:
        
        """ Analyze video file for emotion and age detection """
        # Validate input file
        if file is None:
            raise ValueError('Must have a file to analyze.')

        # Open video file using PyAV
        container = av.open(file, mode="r")
        stream = container.streams.video[0]

        i = 0

        # Process each frame in the video
        for i, frame in enumerate(container.decode(stream)):
            # Convert frame to grayscale for face detection
            gray = frame.to_ndarray()
            # Convert to RGB and then to BGR for age prediction
            frame_rgb = frame.to_rgb().to_ndarray()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Detect faces using Haar cascade
            faces = self.detector.detectMultiScale(gray)

            # Process each detected face
            for face in faces:
                x, y, width, height = face

                # Extract and preprocess face region for emotion detection (grayscale)
                roi_gray = gray[y:y+height, x:x+width]
                roi_gray = ski.transform.resize(roi_gray, (48, 48))
                
                # Extract face region for age prediction (color, BGR format)
                roi_color = frame_bgr[y:y+height, x:x+width]
                roi_color_resized = cv2.resize(roi_color, (227, 227))  # Resize for age model

                # Process frame based on skip setting and valid face region
                if (i % skip == 0) & (np.sum([roi_gray]) != 0):

                    # Preprocess face image for emotion detection model
                    roi = roi_gray.astype('float') / 255.0
                    roi = keras.utils.img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    roi = tf.convert_to_tensor(roi)

                    # Predict emotions using the provided model
                    emotion_prediction = model.predict(roi, verbose=0)[0] > confidence
                    
                    # Predict age distribution using OpenCV DNN
                    age_prediction = self.predict_age(roi_color_resized)> confidence

                    # Store results if confident emotion prediction is found
                    if sum(emotion_prediction > 0):
                        frame_results = pd.Series(
                            np.concatenate([[i],  
                                            emotion_prediction, 
                                            age_prediction,  # Add age prediction results
                                            [x, y, width, height]]), 
                                            index=self.results.columns)
                        self.results = pd.concat([self.results, frame_results.to_frame().T])

            i += 1

        # Close video container
        container.close()

        return True, self.results