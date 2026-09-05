import pandas as pd
import joblib


# 1. Load dataset
df = pd.read_csv("data/payments.csv")


# 2. Load saved ML pipeline
pipeline = joblib.load("models/recovery_pipeline.pkl")


# 3. Prepare features
X = df.drop(columns=["recovered", "transaction_id"])


# 4. Predict recovery probability
df["recovery_probability"] = pipeline.predict_proba(X)[:, 1]


# 5. Calculate expected revenue
df["expected_revenue"] = (
    df["amount"] * df["recovery_probability"]
)


# 6. Define recovery action rules
def recommend_action(row):

    probability = row["recovery_probability"]
    failure_reason = row["failure_reason"]
    retry_count = row["retry_count"]

    # Too many previous retries
    if retry_count >= 3:
        return "DO_NOT_RETRY"

    # Expired card needs payment method update
    if failure_reason == "Expired Card":
        return "UPDATE_PAYMENT_METHOD"

    # Temporary technical/network failures
    if failure_reason in ["Network Error", "Technical Error"]:
        if probability >= 0.50:
            return "RETRY"
        else:
            return "DO_NOT_RETRY"

    # Insufficient funds
    if failure_reason == "Insufficient Funds":
        return "WAIT_AND_NOTIFY"

    # Bank declined
    if failure_reason == "Bank Declined":

        if probability >= 0.60:
            return "RETRY_LATER"
        else:
            return "DO_NOT_RETRY"

    # General probability-based rules
    if probability >= 0.70:
        return "RETRY"

    elif probability >= 0.40:
        return "RETRY_LATER"

    else:
        return "DO_NOT_RETRY"


# 7. Generate recommended actions
df["recommended_action"] = df.apply(
    recommend_action,
    axis=1
)


# 8. Sort by expected revenue
df = df.sort_values(
    by="expected_revenue",
    ascending=False
)


# 9. Display top opportunities
print("\n===== RECOVERY ACTION RECOMMENDATIONS =====\n")

print(
    df[
        [
            "transaction_id",
            "amount",
            "payment_method",
            "failure_reason",
            "retry_count",
            "recovery_probability",
            "expected_revenue",
            "recommended_action"
        ]
    ].head(20).to_string(index=False)
)


# 10. Show action distribution
print("\n===== ACTION DISTRIBUTION =====\n")

print(
    df["recommended_action"].value_counts()
)


# 11. Total expected revenue
total_expected_revenue = df["expected_revenue"].sum()

print("\n======================================")
print(
    "Total expected recoverable revenue: ₹"
    + f"{total_expected_revenue:,.2f}"
)
print("======================================")