import pandas as pd


# 1. Load timing dataset
df = pd.read_csv("data/recovery_timing.csv")


# 2. Calculate historical recovery rate
# for each failure reason and retry delay
timing_rates = (
    df.groupby(
        ["failure_reason", "retry_delay_minutes"]
    )["recovered"]
    .mean()
    .reset_index()
)


# 3. Find the best delay for each failure reason
best_timing = (
    timing_rates
    .sort_values(
        ["failure_reason", "recovered"],
        ascending=[True, False]
    )
    .drop_duplicates(
        "failure_reason"
    )
)


# 4. Rename recovery rate
best_timing = best_timing.rename(
    columns={
        "recovered": "historical_recovery_rate"
    }
)


# 5. Load original payment data
payments = pd.read_csv(
    "data/payments.csv"
)


# 6. Add recommended timing
recommendations = payments.merge(
    best_timing[
        [
            "failure_reason",
            "retry_delay_minutes",
            "historical_recovery_rate"
        ]
    ],
    on="failure_reason",
    how="left"
)


# 7. Rename recommended delay
recommendations = recommendations.rename(
    columns={
        "retry_delay_minutes":
            "recommended_delay_minutes"
    }
)


# 8. Display recommendations
print(
    "\n===== SMART RETRY TIMING RECOMMENDATIONS =====\n"
)

print(
    recommendations[
        [
            "transaction_id",
            "payment_method",
            "failure_reason",
            "retry_count",
            "recommended_delay_minutes",
            "historical_recovery_rate"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# 9. Show timing strategy
print(
    "\n===== TIMING STRATEGY BY FAILURE REASON =====\n"
)

print(
    best_timing[
        [
            "failure_reason",
            "retry_delay_minutes",
            "historical_recovery_rate"
        ]
    ]
    .sort_values("historical_recovery_rate", ascending=False)
    .to_string(index=False)
)


# 10. Show recommendation distribution
print(
    "\n===== RECOMMENDED DELAY DISTRIBUTION =====\n"
)

print(
    recommendations[
        "recommended_delay_minutes"
    ]
    .value_counts()
    .sort_index()
)


# 11. Save recommendations
recommendations.to_csv(
    "data/timing_recommendations.csv",
    index=False
)


print(
    "\nTiming recommendations saved successfully!"
)

print(
    "Location: data/timing_recommendations.csv"
)