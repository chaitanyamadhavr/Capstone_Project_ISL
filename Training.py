import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import average_precision_score

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
    samples = len(os.listdir(os.path.join("Dataset", sign_name)))  # Set the number of samples per sign
    print(samples)
    labels.extend([label_id] * samples)

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, shuffle=True, random_state=42)

# Flatten the sequence dimension to convert 3D features to 2D
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Create a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust the number of trees as needed

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

""" # Calculate mAP for each class
map_scores = []
for i in range(len(CLASSES_LIST)):
    y_true_class = y_test[:, i]
    y_pred_class = y_pred[:, i]
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
plt.show() """

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png')
plt.show()

# Classification Report
class_report = classification_report(y_test, y_pred, target_names=CLASSES_LIST, zero_division=0)
print("Classification Report:\n", class_report)

# Save the trained Random Forest model
joblib.dump(clf, 'random_forest_model.pkl')

# Assuming you have captured the training history as mentioned above
# Random Forest doesn't have loss and accuracy, so you don't need to plot training history.
