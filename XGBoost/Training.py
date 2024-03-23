import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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
# Reshape the features to be 2D
features = features.reshape(-1, 50 * 75 * 3)

# Training
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, shuffle=True)

# Create an XGBoost classifier
clf = xgb.XGBClassifier(
    n_estimators=100,  # Number of boosting rounds (you can adjust this)
    max_depth=3,  # Maximum depth of each tree (you can adjust this)
    learning_rate=0.1,  # Step size shrinkage to prevent overfitting
    objective='multi:softmax',  # For multiclass classification
    num_class=len(CLASSES_LIST)  # Number of classes in your dataset
)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES_LIST, yticklabels=CLASSES_LIST)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig('confusion matrix.png')
plt.show()

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report
class_report = classification_report(y_test, y_pred, target_names=CLASSES_LIST, zero_division=0)
print("Classification Report:\n", class_report)

# Save the trained XGBoost model as a .pkl file
joblib.dump(clf, 'xgboost_model.pkl')
