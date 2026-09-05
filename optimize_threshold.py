import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# 1. Load data and model
# --------------------------------------------------

df = pd.read_csv("data/payments_v2.csv")

pipeline = joblib.load(
    "models/recovery_pipeline_v2.pkl"
)


# --------------------------------------------------
# 2. Prepare data
# --------------------------------------------------

X = df.drop(
    columns=["recovered", "transaction_id", "customer_id"]
)

y = df["recovered"]


# --------------------------------------------------
# 3. Create evaluation split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 4. Get recovery probabilities
# --------------------------------------------------

probabilities = pipeline.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 5. Test different thresholds
# --------------------------------------------------

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


print("\n==============================")
print("THRESHOLD ANALYSIS")
print("==============================")

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
    )