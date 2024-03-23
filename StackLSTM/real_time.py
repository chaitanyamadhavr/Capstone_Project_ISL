#Final LSTM Real Time Testing Code Finalised on 6th October 2023
import numpy as np
import cv2
import os
from tensorflow import keras
import mediapipe as mp
from display import update_window

# Define the class labels
DATA_PATH = "Features"
CLASSES_LIST = os.listdir(DATA_PATH)
print(CLASSES_LIST)

frame_length = 50
slide = 12
threshold = 0.9
eng_text="Hello"

model = keras.models.load_model('model.keras')

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic= mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def draw_landmarks(image, results): 
        # Draw right hand landmarks with red color
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
        )
        
        # Draw left hand landmarks with blue color
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
        )
        
        # Draw shoulder and arm landmarks with green color
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)
        )

# Process Keypoints
def process_keypoints(results):
    pose = [[res.x, res.y, res.z] for res in results.pose_landmarks.landmark] if results.pose_landmarks else [[0.0]*3]*33
    lh = [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else [[0.0]*3]*21
    rh = [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else [[0.0]*3]*21
    landmarks = []
    landmarks.extend(pose)
    landmarks.extend(lh)
    landmarks.extend(rh)
    return landmarks


def detect_landmarks(frame):
    results = holistic.process(frame)
    draw_landmarks(frame,results)
    keypoints = process_keypoints(results)
    return keypoints


# Create a function to preprocess a frame and make predictions
def predict_sign(sequence):
    #Reshape
    sequence = sequence.reshape(1, 50, 225)

    # Make a prediction
    predictions = model.predict(sequence)
    print(predictions)
    if np.max(predictions) >= threshold:
        predicted_class_index = np.argmax(predictions)
        print(predicted_class_index)
        predicted_class = CLASSES_LIST[predicted_class_index]
    else:
        # If no class is predicted with sufficient confidence, set it to "none"
        predicted_class = "none"
    
    return predicted_class

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera (you can change this if you have multiple cameras)
sequence=[]
while cap.isOpened():
    success, frame = cap.read()  # Read a frame from the webcam
    if not success:
        print("Camera got Closed")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = True
    sequence.append(detect_landmarks(frame))

    if(len(sequence)==frame_length):
        eng_text = predict_sign(np.array(sequence))
        sequence = sequence[slide:]
    
    # Display the predicted label on the frame
    cv2.putText(frame, eng_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    update_window(frame,eng_text)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
