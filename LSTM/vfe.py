#Final Feature Extractor Code, Finalised on 6th October 2023
import cv2
import mediapipe as mp
import numpy as np
import os

mp_holistic = mp.solutions.holistic
DATA_PATH = "Dataset"

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

def detect_landmarks(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            success, frame = cap.read()
            if not success:
                print("Scanned", video_path)
                break

            # Make detections
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            keypoints = process_keypoints(results)
            # print(keypoints)
            frames.append(keypoints)
    cap.release()
    return np.array(frames)

#Storage
def create_store_path(sign_name):
    loc = os.path.join("Features", sign_name)
    if not os.path.exists(loc):
        os.makedirs(loc)
    return loc

# Start point
# Get video path
for sign_name in os.listdir(DATA_PATH):
    store_path = create_store_path(sign_name)
    sign_path = os.path.join(DATA_PATH, sign_name)
    i=1
    for video_file in os.listdir(sign_path):
        video_path = os.path.join(sign_path, video_file)
        print("Processing", video_path)
        landmarks_np = detect_landmarks(video_path)
        # print(len(landmarks_np))
        np_store_path=os.path.join(store_path, str(i))
        np.save(np_store_path, landmarks_np)
        i += 1