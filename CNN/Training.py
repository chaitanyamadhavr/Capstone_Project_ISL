import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from model_def import create_cnn_model  # Import the modified CNN model
from sklearn.metrics import classification_report, average_precision_score
import tensorflow as tf
tf.random.set_seed(42)  # Use any integer value as the seed

DATA_PATH = "Selected_Frames"
CLASSES_LIST = os.listdir(DATA_PATH)
print(CLASSES_LIST)

# Load and preprocess the keypoints data
features = []
labels = []

for label_id, sign_name in enumerate(CLASSES_LIST):
    sign_path = os.path.join(DATA_PATH, sign_name)
    np_data = np.load(os.path.join(sign_path, sign_name + ".npy"))
    print("Extracting", sign_name)
    features.extend(np_data)
    samples = len(os.listdir(os.path.join("Dataset", sign_name)))  # Set number of samples per sign
    print(samples)
    labels.extend([label_id] * samples)

# Convert lists to numpy arrays
features = np.array(features)
features = features.reshape(-1, 60, 225)
labels = np.array(to_categorical(labels).astype(int))

# Training
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, shuffle=True)

# Compile the model
input_shape = (60, 225)
num_classes = 61
model = create_cnn_model(input_shape, num_classes)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
history = model.fit(X_train,y_train, epochs=50)  # Adjust the number of epochs as needed
res = model.predict(X_test)

# Calculate mAP for each class
map_scores = []
for i in range(len(CLASSES_LIST)):
    y_true_class = y_test[:, i]
    y_pred_class = res[:, i]
    average_precision = average_precision_score(y_true_class, y_pred_class)
    map_scores.append(average_precision)

# Plot mAP scores
plt.figure(figsize=(8, 6))
plt.bar(CLASSES_LIST, map_scores)
plt.xlabel('Sign Classes')
plt.ylabel('Mean Average Precision (mAP)')
plt.title('Mean Average Precision for Each Sign Class')
plt.xticks(rotation=45)
plt.savefig('MAP.png')
plt.show()

# Generate confusion matrix
y_pred = np.argmax(res, axis=1)
y_true = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig('confusion matrix.png')
plt.show()

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
class_report = classification_report(y_true, y_pred, target_names=CLASSES_LIST, zero_division=0)
print("Classification Report:\n", class_report)

# Assuming you have captured the training history as mentioned above
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['categorical_accuracy'], label='Categorical Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('Training Summary.png')
plt.legend()
plt.show()

# Save the model
model.save('model.keras')
model.summary()