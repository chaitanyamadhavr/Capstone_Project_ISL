""" import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score

DATA_PATH = "Features"
CLASSES_LIST = []

# Load the keypoints data from the .npy files
def load_keypoints(data_path):
    keypoints = np.load(data_path)
    return keypoints

# Load and preprocess the keypoints data
features = []
labels = []

label_id = -1

for sign_name in os.listdir(DATA_PATH):
        CLASSES_LIST.append(sign_name)
        label_id += 1
        sign_path = os.path.join(DATA_PATH, sign_name)
        for video_file in os.listdir(sign_path):
            video_path = os.path.join(sign_path, video_file)
            keypoints_path = os.path.join(video_path, "keypoints.npy")
            print("Extracting ", keypoints_path)
            keypoints = load_keypoints(keypoints_path)
            features.append(keypoints)
            labels.append(label_id)
            
# Convert lists to numpy arrays
features = np.array(features)
print(len(features))
#adjusted_labels = labels - 1
labels = to_categorical(labels).astype(int)
print(len(labels))
print(len(CLASSES_LIST))

#Training
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle= True)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(50,225)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(CLASSES_LIST), activation='softmax'))
print("Model Created Successfully!")

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=200)

res = model.predict(X_test)
model.save('action.h5')
model.save('action.keras')

# Evaluate the trained model.
model_evaluation_history = model.evaluate(X_test, y_test)
 """

import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "Features"
CLASSES_LIST = []

# Load the keypoints data from the .npy files
def load_keypoints(data_path):
    keypoints = np.load(data_path)
    return keypoints

# Load and preprocess the keypoints data
features = []
labels = []

label_id = -1

for sign_name in os.listdir(DATA_PATH):
    CLASSES_LIST.append(sign_name)
    label_id += 1
    sign_path = os.path.join(DATA_PATH, sign_name)
    for video_file in os.listdir(sign_path):
        video_path = os.path.join(sign_path, video_file)
        keypoints_path = os.path.join(video_path, "keypoints.npy")
        print("Extracting ", keypoints_path)
        keypoints = load_keypoints(keypoints_path)
        features.append(keypoints)
        labels.append(label_id)

# Convert lists to numpy arrays
features = np.array(features)
labels = to_categorical(labels).astype(int)

# Training
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, shuffle=True)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(50, 225)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(CLASSES_LIST), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=100)

res = model.predict(X_test)
# Evaluate the trained model.
model_evaluation_history = model.evaluate(X_test, y_test)
# Generate confusion matrix
y_pred = np.argmax(res, axis=1)
y_true = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Save the model
model.save('action.h5')
model.save('action.keras')


