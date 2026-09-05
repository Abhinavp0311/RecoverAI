import pandas as pd
import joblib


# 1. Load dataset
df = pd.read_csv("data/payments.csv")


# 2. Load saved ML pipeline
pipeline = joblib.load("models/recovery_pipeline.pkl")


# 3. Separate features
X = df.drop(columns=["recovered", "transaction_id"])


# 4. Get recovery probabilities
probabilities = pipeline.predict_proba(X)[:, 1]


# 5. Create predictions table
predictions = df.drop(columns=["recovered"]).copy()

predictions["recovery_probability"] = probabilities


# 6. Calculate expected recoverable revenue
predictions["expected_revenue"] = (
    predictions["amount"] *
    predictions["recovery_probability"]
)


# 7. Calculate recovery priority
predictions["priority_score"] = (
    predictions["expected_revenue"] *
    (1 - predictions["retry_count"] / 5)
)


# 8. Sort by priority
predictions = predictions.sort_values(
    by="priority_score",
    ascending=False
)


# 9. Display top 20 payments
print("\n===== TOP RECOVERY OPPORTUNITIES =====\n")

print(
    predictions[
        [
            "transaction_id",
            "amount",
            "payment_method",
            "failure_reason",
            "retry_count",
            "recovery_probability",
            "expected_revenue",
            "priority_score"
        ]
    ].head(20).to_string(index=False)
)


# 10. Calculate total expected recoverable revenue
total_expected_revenue = predictions["expected_revenue"].sum()

print("\n======================================")
print(
    "Total expected recoverable revenue: ₹"
    + f"{total_expected_revenue:,.2f}"
)
print("======================================")