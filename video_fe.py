'''Edited on 18th august 2023'''
import cv2
import mediapipe as mp
import numpy as np
import os

frame_length = 50

mp_holistic = mp.solutions.holistic

# Process Keypoints
def process_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])


def save_keypoints(video_directory, keypoints_list):
    npy_path = os.path.join(video_directory, "keypoints.npy")
    # print(np.array(keypoints_list))
    np.save(npy_path, np.array(keypoints_list))

def create_store_path(sign_name, video_file):
    loc = os.path.join("Features", sign_name, os.path.splitext(video_file)[0])
    if not os.path.exists(loc):
        os.makedirs(loc)
    return loc

# Start point
if __name__ == "__main__":

    DATA_PATH = os.path.join('Emotions')

    # Get video path
    for sign_name in os.listdir(DATA_PATH):
        sign_path = os.path.join(DATA_PATH, sign_name)
        for video_file in os.listdir(sign_path):
            video_path = os.path.join(sign_path, video_file)
            print("Processing", video_path)
            store_path = create_store_path(sign_name, video_file)

            keypoints_list = []

            cap = cv2.VideoCapture(video_path)
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
                skip_frame = frame_count//frame_length

                if skip_frame<1:
                        print("Frames are less than", frame_length)
                        break
                
                frame_number = 1
                count = 1

                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        print("Done with", video_path)
                        break

                    if frame_number % skip_frame == 0: # Select every Xth frame
                        if count>frame_length:
                            print("Captured", frame_length)
                            break  
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # cv2.imshow("Hand Detection", frame)
                        results = holistic.process(frame)
                        keypoints = process_keypoints(results)
                        # print(len(keypoints))
                        keypoints_list.append(keypoints)
                        count += 1
                        
                    frame_number += 1
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
            
            # print(len(keypoints_list))

            save_keypoints(store_path, keypoints_list)
            cap.release()
