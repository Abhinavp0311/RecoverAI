import numpy as np
import pandas as pd


# ==========================================
# 1. Reproducibility
# ==========================================

np.random.seed(42)

n = 15000
n_customers = 3000


# ==========================================
# 2. Create customers
# ==========================================

customer_ids = np.arange(
    10000,
    10000 + n_customers
)

customer_success_rate = np.clip(
    np.random.beta(7, 3, n_customers),
    0.05,
    0.98
)

customer_lifetime_value = np.random.lognormal(
    mean=9.0,
    sigma=0.8,
    size=n_customers
)


customers = pd.DataFrame({
    "customer_id": customer_ids,
    "customer_success_rate": customer_success_rate,
    "customer_lifetime_value": customer_lifetime_value
})


# ==========================================
# 3. Generate transactions
# ==========================================

transaction_customer = np.random.choice(
    customer_ids,
    size=n
)

df = pd.DataFrame({
    "transaction_id": np.arange(1, n + 1),
    "customer_id": transaction_customer
})


# ==========================================
# 4. Attach customer information
# ==========================================

df = df.merge(
    customers,
    on="customer_id",
    how="left"
)


# ==========================================
# 5. Payment information
# ==========================================

df["amount"] = np.random.lognormal(
    mean=8.0,
    sigma=1.0,
    size=n
)

df["amount"] = df["amount"].clip(
    50,
    200000
)


df["payment_method"] = np.random.choice(
    [
        "UPI",
        "Card",
        "NetBanking",
        "Wallet"
    ],
    size=n,
    p=[
        0.45,
        0.30,
        0.15,
        0.10
    ]
)


df["failure_reason"] = np.random.choice(
    [
        "Insufficient Funds",
        "Bank Declined",
        "Network Error",
        "Technical Error",
        "Expired Card"
    ],
    size=n,
    p=[
        0.25,
        0.20,
        0.20,
        0.20,
        0.15
    ]
)


df["retry_count"] = np.random.poisson(
    1.2,
    size=n
)

df["retry_count"] = df[
    "retry_count"
].clip(0, 5)


df["transaction_hour"] = np.random.randint(
    0,
    24,
    size=n
)


# ==========================================
# 6. Create realistic recovery signal
# ==========================================

logit = np.full(n, -0.8)


# Customer payment history
logit += (
    2.2 *
    (df["customer_success_rate"] - 0.5)
)


# Retry penalty
logit -= (
    0.35 *
    df["retry_count"]
)


# Failure reason effects
logit += np.where(
    df["failure_reason"] == "Network Error",
    1.0,
    0
)

logit += np.where(
    df["failure_reason"] == "Technical Error",
    0.7,
    0
)

logit += np.where(
    df["failure_reason"] == "Bank Declined",
    -0.2,
    0
)

logit += np.where(
    df["failure_reason"] == "Insufficient Funds",
    -0.6,
    0
)

logit += np.where(
    df["failure_reason"] == "Expired Card",
    -1.2,
    0
)


# Payment method effects
logit += np.where(
    df["payment_method"] == "UPI",
    0.1,
    0
)

logit += np.where(
    df["payment_method"] == "Wallet",
    -0.15,
    0
)


# Transaction amount effect
logit -= (
    0.000002 *
    df["amount"]
)


# Some time-of-day variation
logit += np.where(
    df["transaction_hour"].between(
        9,
        18
    ),
    0.15,
    -0.05
)


# Random noise
logit += np.random.normal(
    0,
    0.6,
    size=n
)


# Convert logit to probability
recovery_probability = (
    1 /
    (
        1 +
        np.exp(-logit)
    )
)


# ==========================================
# 7. Generate recovery outcome
# ==========================================

df["recovered"] = np.random.binomial(
    1,
    recovery_probability
)


# ==========================================
# 8. Remove hidden simulation probability
# ==========================================




# ==========================================
# 9. Save dataset
# ==========================================

df.to_csv(
    "data/payments_v2.csv",
    index=False
)


# ==========================================
# 10. Summary
# ==========================================

print(
    "Realistic synthetic dataset created!"
)

print(
    f"Rows: {len(df)}"
)

print(
    "\nColumns:"
)

print(
    df.columns.tolist()
)

print(
    "\nRecovery distribution:"
)

print(
    df["recovered"].value_counts()
)

print(
    "\nRecovery rate:"
)

print(
    round(
        df["recovered"].mean(),
        4
    )
)

print(
    "\nSaved to:"
)

print(
    "data/payments_v2.csv"
)