import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# --------------------------------------------------
# 1. Load realistic synthetic dataset
# --------------------------------------------------

df = pd.read_csv("data/payments_v2.csv")

print("Dataset loaded!")
print("Shape:", df.shape)


# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["recovered", "transaction_id", "customer_id"])
y = df["recovered"]


# --------------------------------------------------
# 3. Train / test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# --------------------------------------------------
# 4. Define feature types
# --------------------------------------------------

numerical_features = [
    "customer_success_rate",
    "customer_lifetime_value",
    "amount",
    "retry_count",
    "transaction_hour"
]

categorical_features = [
    "payment_method",
    "failure_reason"
]


# --------------------------------------------------
# 5. Preprocessing
# --------------------------------------------------

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


# --------------------------------------------------
# 6. Logistic Regression model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# --------------------------------------------------
# 7. Create ML pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 8. Train
# --------------------------------------------------

print("\nTraining Model V2...")

pipeline.fit(X_train, y_train)

print("Training complete!")


# --------------------------------------------------
# 9. Predictions
# --------------------------------------------------

predictions = pipeline.predict(X_test)

probabilities = pipeline.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 10. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)


print("\n==============================")
print("MODEL V2 RESULTS")
print("==============================")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


# --------------------------------------------------
# 11. Save Model V2
# --------------------------------------------------

model_path = "models/recovery_pipeline_v2.pkl"

joblib.dump(
    pipeline,
    model_path
)

print("\nModel saved to:")
print(model_path)