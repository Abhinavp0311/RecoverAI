import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# --------------------------------------------------
# 1. Load data and model
# --------------------------------------------------

df = pd.read_csv("data/payments_v2.csv")

pipeline = joblib.load(
    "models/recovery_pipeline_v2.pkl"
)


# --------------------------------------------------
# 2. Prepare features
# --------------------------------------------------

X = df.drop(
    columns=["recovered", "transaction_id", "customer_id"]
)

y = df["recovered"]

groups = df["customer_id"]


# --------------------------------------------------
# 3. Split by customer
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]


print("Total transactions:", len(df))
print("Test transactions:", len(X_test))
print(
    "Unique customers in test:",
    groups.iloc[test_idx].nunique()
)


# --------------------------------------------------
# 4. Evaluate
# --------------------------------------------------

predictions = pipeline.predict(X_test)

probabilities = pipeline.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


# --------------------------------------------------
# 5. Results
# --------------------------------------------------

print("\n==============================")
print("GROUPED MODEL V2 RESULTS")
print("==============================")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")