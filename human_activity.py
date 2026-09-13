# ============================================================
# HUMAN ACTIVITY RECOGNITION USING SMARTPHONES
# Machine Learning Project - Random Forest
# ============================================================

# -------------------- 1. IMPORT LIBRARIES --------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# -------------------- 2. DATASET PATH ------------------------

DATASET_PATH = "UCI HAR Dataset"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH = os.path.join(DATASET_PATH, "test")


# -------------------- 3. CHECK DATASET ------------------------

if not os.path.exists(DATASET_PATH):
    print("ERROR: 'UCI HAR Dataset' folder not found!")
    print("Please extract the ZIP file and place the folder")
    print("in the same location as this Python program.")
    exit()

print("=" * 60)
print(" HUMAN ACTIVITY RECOGNITION USING SMARTPHONES")
print("=" * 60)


# -------------------- 4. LOAD FEATURES ------------------------

features = pd.read_csv(
    os.path.join(DATASET_PATH, "features.txt"),
    sep=r"\s+",
    header=None
)

feature_names = features[1].values

print("\nNumber of Features:", len(feature_names))


# -------------------- 5. LOAD TRAINING DATA -------------------

X_train = pd.read_csv(
    os.path.join(TRAIN_PATH, "X_train.txt"),
    sep=r"\s+",
    header=None
)

y_train = pd.read_csv(
    os.path.join(TRAIN_PATH, "y_train.txt"),
    sep=r"\s+",
    header=None
)


# -------------------- 6. LOAD TESTING DATA --------------------

X_test = pd.read_csv(
    os.path.join(TEST_PATH, "X_test.txt"),
    sep=r"\s+",
    header=None
)

y_test = pd.read_csv(
    os.path.join(TEST_PATH, "y_test.txt"),
    sep=r"\s+",
    header=None
)


# -------------------- 7. ASSIGN COLUMN NAMES -----------------

X_train.columns = feature_names
X_test.columns = feature_names


# -------------------- 8. REMOVE DUPLICATE COLUMNS -----------

# Some versions of the dataset can contain duplicate
# feature names. Random Forest can work with them, but
# removing duplicate columns makes the dataset cleaner.

X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]


# -------------------- 9. LOAD ACTIVITY LABELS ----------------

activity_labels = pd.read_csv(
    os.path.join(DATASET_PATH, "activity_labels.txt"),
    sep=r"\s+",
    header=None
)

activity_labels.columns = ["ID", "Activity"]

activity_dict = dict(
    zip(
        activity_labels["ID"],
        activity_labels["Activity"]
    )
)


# -------------------- 10. CONVERT LABELS ---------------------

y_train = y_train[0].map(activity_dict)
y_test = y_test[0].map(activity_dict)


# -------------------- 11. DISPLAY DATA INFORMATION -----------

print("\n" + "=" * 60)
print(" DATASET INFORMATION")
print("=" * 60)

print("\nTraining samples :", X_train.shape[0])
print("Training features:", X_train.shape[1])

print("\nTesting samples  :", X_test.shape[0])
print("Testing features :", X_test.shape[1])

print("\nActivities:")
for activity in activity_labels["Activity"]:
    print("-", activity)


# -------------------- 12. CHECK MISSING VALUES ---------------

print("\n" + "=" * 60)
print(" MISSING VALUE CHECK")
print("=" * 60)

missing_train = X_train.isnull().sum().sum()
missing_test = X_test.isnull().sum().sum()

print("Missing values in training data:", missing_train)
print("Missing values in testing data :", missing_test)


# -------------------- 13. TRAIN RANDOM FOREST -----------------

print("\n" + "=" * 60)
print(" MODEL TRAINING")
print("=" * 60)

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed successfully!")


# -------------------- 14. MAKE PREDICTIONS -------------------

print("\nMaking predictions...")

y_pred = model.predict(X_test)

print("Prediction completed!")


# -------------------- 15. CALCULATE ACCURACY ------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print(" MODEL ACCURACY")
print("=" * 60)

print("\nAccuracy:", accuracy)
print("Accuracy Percentage: {:.2f}%".format(accuracy * 100))


# -------------------- 16. CLASSIFICATION REPORT ---------------

print("\n" + "=" * 60)
print(" CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# -------------------- 17. CONFUSION MATRIX --------------------

print("\n" + "=" * 60)
print(" CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=activity_labels["Activity"].values
)

print("\n")
print(cm)


# -------------------- 18. DISPLAY CONFUSION MATRIX -----------

plt.figure(figsize=(10, 8))
plt.savefig("confusion_matrix.png", dpi=300)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=activity_labels["Activity"].values
)

disp.plot(
    xticks_rotation=45,
    values_format="d"
)

plt.title("Human Activity Recognition - Confusion Matrix")
plt.tight_layout()
plt.show()


# -------------------- 19. ACTIVITY DISTRIBUTION ---------------

plt.figure(figsize=(10, 6))

y_train.value_counts().plot(
    kind="bar"
)

plt.title("Training Dataset - Activity Distribution")
plt.xlabel("Activity")
plt.ylabel("Number of Samples")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# -------------------- 20. FEATURE IMPORTANCE ------------------

print("\n" + "=" * 60)
print(" TOP 10 IMPORTANT FEATURES")
print("=" * 60)

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(
    feature_importance.head(10).to_string(
        index=False
    )
)


# -------------------- 21. FEATURE IMPORTANCE GRAPH ------------

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# -------------------- 22. SAMPLE PREDICTION -------------------

print("\n" + "=" * 60)
print(" SAMPLE ACTIVITY PREDICTION")
print("=" * 60)

sample_number = 0

sample = X_test.iloc[
    sample_number
].values.reshape(1, -1)

prediction = model.predict(sample)[0]

actual = y_test.iloc[
    sample_number
]

print("\nSample Number:", sample_number + 1)
print("Predicted Activity:", prediction)
print("Actual Activity   :", actual)

if prediction == actual:
    print("Result: CORRECT PREDICTION")
else:
    print("Result: INCORRECT PREDICTION")


# -------------------- 23. MULTIPLE SAMPLE PREDICTIONS --------

print("\n" + "=" * 60)
print(" FIRST 10 PREDICTIONS")
print("=" * 60)

results = pd.DataFrame({
    "Actual Activity": y_test.iloc[:10].values,
    "Predicted Activity": y_pred[:10]
})

print(results.to_string(index=False))


# -------------------- 24. FINAL RESULT ------------------------

print("\n" + "=" * 60)
print(" PROJECT COMPLETED")
print("=" * 60)

print("\nProject Title:")
print("Human Activity Recognition Using Smartphones")

print("\nMachine Learning Algorithm:")
print("Random Forest Classifier")

print("\nFinal Accuracy: {:.2f}%".format(
    accuracy * 100
))

print("\nActivities Recognized:")

for activity in activity_labels["Activity"]:
    print("✓", activity)

print("\nThank you!")
print("=" * 60)
