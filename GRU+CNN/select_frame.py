#Final select frames code, finalised on 6th October 2023
import numpy as np
import os


frame_length = 60
DATA_PATH = "Features"


# Frames Selection
def select_frames(features):
    selected = []
    frame_count = len(features)
    skip_frame = frame_count//frame_length
    if skip_frame<1:
        print("Frames are less than", frame_length)
        exit(1)
    rem_frame = frame_count%frame_length
    s = rem_frame//2
    if(s!=0):
        features = features[s:-s]
    for i in range(frame_length):
        selected.append(features[i*skip_frame])
    return selected


#Storage
def create_store_path(sign_name):
    loc = os.path.join("Selected_Frames", sign_name)
    if not os.path.exists(loc):
        os.makedirs(loc)
    return loc


# Get Features path
for sign_name in os.listdir(DATA_PATH):
    selected_features = []
    store_path = create_store_path(sign_name)
    sign_path = os.path.join(DATA_PATH, sign_name)
    for feature_file in os.listdir(sign_path):
        feature_path = os.path.join(sign_path, feature_file)
        print("Processing", feature_path)
        selected = select_frames(np.load(feature_path))
        selected_features.append(selected)
    np_store_path=os.path.join(store_path, sign_name)
    np.save(np_store_path, np.array(selected_features))
