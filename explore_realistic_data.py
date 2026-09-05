import pandas as pd


# Load realistic synthetic dataset
df = pd.read_csv("data/payments_v2.csv")


print("===== DATASET OVERVIEW =====")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


# Target distribution
print("\n===== RECOVERY DISTRIBUTION =====")

print(
    df["recovered"].value_counts()
)

print("\nRecovery rate:")
print(
    round(df["recovered"].mean(), 4)
)


# Recovery by payment method
print(
    "\n===== RECOVERY BY PAYMENT METHOD ====="
)

print(
    df.groupby("payment_method")["recovered"]
    .mean()
    .sort_values(ascending=False)
)


# Recovery by failure reason
print(
    "\n===== RECOVERY BY FAILURE REASON ====="
)

print(
    df.groupby("failure_reason")["recovered"]
    .mean()
    .sort_values(ascending=False)
)


# Recovery by retry count
print(
    "\n===== RECOVERY BY RETRY COUNT ====="
)

print(
    df.groupby("retry_count")["recovered"]
    .mean()
    .sort_index()
)


# Numerical summary
print(
    "\n===== NUMERICAL FEATURES ====="
)

print(
    df[
        [
            "amount",
            "retry_count",
            "transaction_hour",
            "customer_lifetime_value"
        ]
    ].describe()
)