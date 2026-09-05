import numpy as np
import pandas as pd


# Reproducibility
np.random.seed(42)

n = 10000


# -----------------------------
# Basic payment information
# -----------------------------

payment_method = np.random.choice(
    ["UPI", "Card", "NetBanking", "Wallet"],
    size=n
)

failure_reason = np.random.choice(
    [
        "Insufficient Funds",
        "Bank Declined",
        "Network Error",
        "Technical Error",
        "Expired Card"
    ],
    size=n
)

retry_count = np.random.randint(0, 5, size=n)

transaction_hour = np.random.randint(0, 24, size=n)


# -----------------------------
# Simulated recovery timing
# -----------------------------

# Possible retry delays
retry_delay_options = [
    0,      # immediately
    15,     # 15 minutes
    60,     # 1 hour
    360,    # 6 hours
    1440    # 24 hours
]


retry_delay_minutes = np.random.choice(
    retry_delay_options,
    size=n
)


# -----------------------------
# Simulate probability of
# successful recovery
# -----------------------------

timing_score = np.full(n, 0.40)


# Network and technical errors
# tend to recover faster
timing_score += np.where(
    failure_reason == "Network Error",
    0.25,
    0
)

timing_score += np.where(
    failure_reason == "Technical Error",
    0.20,
    0
)


# Insufficient funds may benefit
# from waiting
timing_score += np.where(
    failure_reason == "Insufficient Funds",
    0.10,
    0
)


# Bank declines are less predictable
timing_score += np.where(
    failure_reason == "Bank Declined",
    -0.05,
    0
)


# Expired cards are unlikely to recover
# simply by retrying
timing_score += np.where(
    failure_reason == "Expired Card",
    -0.20,
    0
)


# Too many retries reduce success
timing_score -= retry_count * 0.05


# Certain retry delays work better
# depending on failure type

timing_score += np.where(
    (failure_reason == "Network Error") &
    (retry_delay_minutes == 15),
    0.15,
    0
)

timing_score += np.where(
    (failure_reason == "Technical Error") &
    (retry_delay_minutes == 60),
    0.15,
    0
)

timing_score += np.where(
    (failure_reason == "Insufficient Funds") &
    (retry_delay_minutes == 1440),
    0.20,
    0
)

timing_score += np.where(
    (failure_reason == "Bank Declined") &
    (retry_delay_minutes == 360),
    0.10,
    0
)


# Add randomness so the model
# does not get a perfect dataset
timing_score += np.random.normal(
    0,
    0.08,
    size=n
)


# Keep probabilities between 0 and 1
timing_score = np.clip(
    timing_score,
    0.05,
    0.95
)


# Generate recovery outcome
recovered = np.random.binomial(
    1,
    timing_score
)


# -----------------------------
# Create dataframe
# -----------------------------

df = pd.DataFrame({
    "payment_method": payment_method,
    "failure_reason": failure_reason,
    "retry_count": retry_count,
    "transaction_hour": transaction_hour,
    "retry_delay_minutes": retry_delay_minutes,
    "recovered": recovered
})


# -----------------------------
# Save dataset
# -----------------------------

df.to_csv(
    "data/recovery_timing.csv",
    index=False
)


print("Timing dataset created successfully!")
print(f"Rows: {len(df)}")
print("\nColumns:")
print(df.columns.tolist())

print("\nRecovery distribution:")
print(df["recovered"].value_counts())

print("\nSaved to:")
print("data/recovery_timing.csv")