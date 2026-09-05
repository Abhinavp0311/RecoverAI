import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# 1. Load timing dataset
df = pd.read_csv("data/recovery_timing.csv")


# 2. Separate features and target
X = df.drop(columns=["recovered"])
y = df["recovered"]


# 3. Define feature types
numerical_features = [
    "retry_count",
    "transaction_hour"
]

categorical_features = [
    "payment_method",
    "failure_reason",
    "retry_delay_minutes"
]


# 4. Create preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# 5. Create model pipeline
pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 7. Train model
pipeline.fit(X_train, y_train)


# 8. Predictions
y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# 9. Evaluate model
print("\n===== TIMING MODEL PERFORMANCE =====")

print(
    "Accuracy :",
    round(accuracy_score(y_test, y_pred), 4)
)

print(
    "Precision:",
    round(precision_score(y_test, y_pred), 4)
)

print(
    "Recall   :",
    round(recall_score(y_test, y_pred), 4)
)

print(
    "F1 Score :",
    round(f1_score(y_test, y_pred), 4)
)

print(
    "ROC-AUC  :",
    round(roc_auc_score(y_test, y_probability), 4)
)


# 10. Save model
joblib.dump(
    pipeline,
    "models/timing_pipeline.pkl"
)


print("\nTiming model saved successfully!")

print(
    "Location: models/timing_pipeline.pkl"
)