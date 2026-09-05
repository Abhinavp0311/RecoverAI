import pandas as pd
import joblib


# ==========================================
# 1. Load data and trained recovery model
# ==========================================

payments = pd.read_csv("data/payments.csv")

recovery_pipeline = joblib.load(
    "models/recovery_pipeline.pkl"
)


# ==========================================
# 2. Predict recovery probability
# ==========================================

X = payments.drop(
    columns=["recovered", "transaction_id"]
)

payments["recovery_probability"] = (
    recovery_pipeline.predict_proba(X)[:, 1]
)


# ==========================================
# 3. Calculate expected revenue
# ==========================================

payments["expected_revenue"] = (
    payments["amount"]
    * payments["recovery_probability"]
)


# ==========================================
# 4. Calculate recommended timing
# ==========================================

timing_data = pd.read_csv(
    "data/recovery_timing.csv"
)

timing_rates = (
    timing_data
    .groupby(
        ["failure_reason", "retry_delay_minutes"]
    )["recovered"]
    .mean()
    .reset_index()
)

best_timing = (
    timing_rates
    .sort_values(
        ["failure_reason", "recovered"],
        ascending=[True, False]
    )
    .drop_duplicates("failure_reason")
)

best_timing = best_timing.rename(
    columns={
        "retry_delay_minutes":
            "recommended_retry_delay",
        "recovered":
            "historical_timing_recovery_rate"
    }
)


payments = payments.merge(
    best_timing[
        [
            "failure_reason",
            "recommended_retry_delay",
            "historical_timing_recovery_rate"
        ]
    ],
    on="failure_reason",
    how="left"
)


# ==========================================
# 5. Final recovery action
# ==========================================

def decide_action(row):

    probability = row["recovery_probability"]
    failure_reason = row["failure_reason"]
    retry_count = row["retry_count"]

    # Too many retries
    if retry_count >= 3:
        return "DO_NOT_RETRY"

    # Payment method needs fixing
    if failure_reason == "Expired Card":
        return "UPDATE_PAYMENT_METHOD"

    # Insufficient funds
    if failure_reason == "Insufficient Funds":
        return "WAIT_AND_NOTIFY"

    # Network or technical failures
    if failure_reason in [
        "Network Error",
        "Technical Error"
    ]:

        if probability >= 0.50:
            return "RETRY"

        return "DO_NOT_RETRY"

    # Bank declined
    if failure_reason == "Bank Declined":

        if probability >= 0.60:
            return "RETRY_LATER"

        return "DO_NOT_RETRY"

    # General probability rules
    if probability >= 0.70:
        return "RETRY"

    if probability >= 0.40:
        return "RETRY_LATER"

    return "DO_NOT_RETRY"


payments["recommended_action"] = (
    payments.apply(
        decide_action,
        axis=1
    )
)


# ==========================================
# 6. Remove retry timing when we shouldn't
# retry
# ==========================================

payments["final_retry_delay_minutes"] = (
    payments["recommended_retry_delay"]
)

payments.loc[
    payments["recommended_action"].isin(
        [
            "DO_NOT_RETRY",
            "UPDATE_PAYMENT_METHOD",
            "WAIT_AND_NOTIFY"
        ]
    ),
    "final_retry_delay_minutes"
] = None


# ==========================================
# 7. Calculate priority score
# ==========================================

payments["priority_score"] = (
    payments["expected_revenue"]
    * (1 - payments["retry_count"] / 5)
)


# ==========================================
# 8. Sort highest-value opportunities first
# ==========================================

payments = payments.sort_values(
    by="priority_score",
    ascending=False
)


# ==========================================
# 9. Display final decisions
# ==========================================

print(
    "\n===== RECOVERAI FINAL DECISION ENGINE =====\n"
)

print(
    payments[
        [
            "transaction_id",
            "amount",
            "failure_reason",
            "retry_count",
            "recovery_probability",
            "expected_revenue",
            "recommended_action",
            "final_retry_delay_minutes",
            "priority_score"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ==========================================
# 10. Action distribution
# ==========================================

print(
    "\n===== ACTION DISTRIBUTION =====\n"
)

print(
    payments[
        "recommended_action"
    ].value_counts()
)


# ==========================================
# 11. Timing distribution
# ==========================================

print(
    "\n===== FINAL RETRY TIMING DISTRIBUTION =====\n"
)

print(
    payments[
        "final_retry_delay_minutes"
    ]
    .value_counts(dropna=False)
    .sort_index()
)


# ==========================================
# 12. Save final decisions
# ==========================================

payments.to_csv(
    "data/recoverai_decisions.csv",
    index=False
)


print(
    "\nFinal RecoverAI decisions saved successfully!"
)

print(
    "Location: data/recoverai_decisions.csv"
)