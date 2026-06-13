import pandas as pd
import numpy as np
import plotly.express as px
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# 1️ Load Dataset

data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Total rows in dataset:", len(data))


# 2️ Data Cleaning

# Convert TotalCharges to numeric (fix blank spaces)
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

# Fill missing values with median
data["TotalCharges"] = data["TotalCharges"].fillna(
    data["TotalCharges"].median()
)

# Drop customerID (not useful for prediction)
data.drop("customerID", axis=1, inplace=True)

# 3️ Encode Categorical Columns

label_encoders = {}

for column in data.columns:
    if data[column].dtype == "object":
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# 4️ Split Features & Target

X = data.drop("Churn", axis=1)
y = data["Churn"]

print("\nChurn Distribution:")
print(y.value_counts())


# 5️ Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining set size:", len(X_train))
print("Test set size:", len(X_test))

# 6️ Train Model

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# 7️⃣ Evaluate Model

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print("\nTest Accuracy: {:.2f}%".format(accuracy * 100))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8️⃣ Cross Validation

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print(
"\nCross Validation Accuracy: {:.2f}%".format(
cv_scores.mean() * 100
)
)

# Save metrics

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "cv_accuracy": cv_scores.mean()
}

pickle.dump(
    metrics,
    open("model_metrics.pkl", "wb")
)

pickle.dump(
cm,
open("confusion_matrix.pkl", "wb")
)

feature_importance = pd.DataFrame({
"Feature": X.columns,
"Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
by="Importance",
ascending=False
)

pickle.dump(
feature_importance,
open("feature_importance.pkl", "wb")
)


print("feature_importance.pkl created successfully")

print("\nFiles saved successfully!")
print("model_metrics.pkl")
print("confusion_matrix.pkl")
print("feature_importance.pkl")

# 9️⃣ Save Everything for Deployment

# Save model

pickle.dump(
    model,
    open("churn_model.pkl", "wb")
)

# Save label encoders

pickle.dump(
    label_encoders,
    open("label_encoders.pkl", "wb")
)

# Save feature column order

pickle.dump(
    X.columns,
    open("feature_columns.pkl", "wb")
)

print("\nModel, Encoders, and Feature Columns saved successfully!")
