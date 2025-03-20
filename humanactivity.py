import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. LOADING THE DATA ---
PATH = "UCI HAR Dataset/"
features_path = PATH + "features.txt"
activity_labels_path = PATH + "activity_labels.txt"

X_train_path = PATH + "train/X_train.txt"
y_train_path = PATH + "train/y_train.txt"
subject_train_path = PATH + "train/subject_train.txt"

X_test_path = PATH + "test/X_test.txt"
y_test_path = PATH + "test/Y_test.txt"
subject_test_path = PATH + "test/subject_test.txt"

# Load feature names
features_df = pd.read_csv(features_path, sep="\\s+", header=None, names=["idx", "feature"])
feature_names = features_df["feature"].tolist()

# Check for duplicates in the feature names
if len(feature_names) != len(set(feature_names)):
    print("Warning: Duplicate feature names detected!")

# Modify the feature names to ensure uniqueness
feature_name_count = {}
for i, feature in enumerate(feature_names):
    if feature in feature_name_count:
        feature_name_count[feature] += 1
        # Append a suffix to duplicates
        feature_names[i] = f"{feature}_{feature_name_count[feature]}"
    else:
        feature_name_count[feature] = 0

# Load activity labels (mapping IDs 1-6 to string names)
activity_labels_df = pd.read_csv(activity_labels_path, sep="\\s+", header=None, names=["id", "activity"])
activity_map = dict(zip(activity_labels_df.id, activity_labels_df.activity))

# Load the train and test sets
X_train = pd.read_csv(X_train_path, sep="\\s+", header=None, names=feature_names)
y_train = pd.read_csv(y_train_path, sep="\\s+", header=None, names=["Activity"])
subject_train = pd.read_csv(subject_train_path, sep="\\s+", header=None, names=["Subject"])

X_test = pd.read_csv(X_test_path, sep="\\s+", header=None, names=feature_names)
y_test = pd.read_csv(y_test_path, sep="\\s+", header=None, names=["Activity"])
subject_test = pd.read_csv(subject_test_path, sep="\\s+", header=None, names=["Subject"])

# Map the activity IDs to their names
y_train["Activity"] = y_train["Activity"].map(activity_map)
y_test["Activity"] = y_test["Activity"].map(activity_map)

# --- 2. CONVERT MULTI-CLASS TO BINARY ---
def to_binary_label(activity):
    if activity in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]:
        return 1  # Active
    else:
        return 0  # Inactive

y_train["Binary"] = y_train["Activity"].apply(to_binary_label)
y_test["Binary"] = y_test["Activity"].apply(to_binary_label)

# --- 3. COMBINE FEATURES AND SUBJECT INFORMATION ---
# Add subject IDs to the features
X_train["Subject"] = subject_train["Subject"]
X_test["Subject"] = subject_test["Subject"]

# --- 4. SPLIT THE DATA INTO TRAIN AND TEST (WITHOUT SUBJECT-WISE SPLIT) ---
# Combine the features and labels, without subject-wise split
train_data = pd.concat([X_train, y_train["Binary"], subject_train], axis=1)
test_data = pd.concat([X_test, y_test["Binary"], subject_test], axis=1)

# Split features and labels
X_train_final = train_data.drop(columns=["Binary", "Subject"])
y_train_final = train_data["Binary"]

X_test_final = test_data.drop(columns=["Binary", "Subject"])
y_test_final = test_data["Binary"]

# --- 5. TRAIN THE SVM MODELS WITH PCA FOR DIMENSIONALITY REDUCTION ---
scaler = StandardScaler()

# Define a pipeline that includes scaling, PCA, and SVM with class balancing
pipe = Pipeline([
    ('scaler', scaler),  # Standardize the features
    ('pca', PCA(n_components=50)),  # Reduce dimensionality from 561 to 50
    ('svc', SVC(class_weight='balanced'))  # SVM with class_weight='balanced' for handling class imbalance
])

# --- 6. HYPERPARAMETER TUNING WITH GRIDSEARCHCV FOR KERNEL PARAMETERS ---
param_grid = [
    {
        'svc__kernel': ['linear'],
        'svc__C': [0.1, 1, 10, 100]
    },
    {
        'svc__kernel': ['poly'],
        'svc__C': [0.1, 1],
        'svc__degree': [2, 3],
        'svc__gamma': [0.001, 0.01, 0.1]
    },
    {
        'svc__kernel': ['rbf'],
        'svc__C': [0.1, 1, 10],
        'svc__gamma': [0.001, 0.01, 0.1]
    }
]

# GridSearchCV to find the best parameters for the model
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='accuracy',  # Evaluate using accuracy, or you can choose 'f1', 'precision', or 'recall'
    cv=3,  # 3-fold cross-validation
    n_jobs=-1,  # Use all processors for parallel computation
    verbose=1
)

# Fit the model with GridSearchCV
grid_search.fit(X_train_final, y_train_final)

# Output the best parameters found by GridSearchCV
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# --- 7. MODEL EVALUATION ---
y_pred = grid_search.best_estimator_.predict(X_test_final)

print("Confusion Matrix:")
print(confusion_matrix(y_test_final, y_pred))

print("Classification Report:")
print(classification_report(y_test_final, y_pred))
