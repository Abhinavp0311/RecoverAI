import pandas as pd


# -----------------------------------------
# 1. Load the dataset
# -----------------------------------------

df = pd.read_csv("data/payments.csv")


# -----------------------------------------
# 2. Basic dataset information
# -----------------------------------------

print("===================================")
print("RecoverAI Dataset Exploration")
print("===================================")

print()

print("Dataset shape:")
print(df.shape)

print()

print("Columns:")
print(df.columns.tolist())

print()


# -----------------------------------------
# 3. First few records
# -----------------------------------------

print("First 5 records:")
print(df.head())

print()


# -----------------------------------------
# 4. Data types
# -----------------------------------------

print("Data types:")
print(df.dtypes)

print()


# -----------------------------------------
# 5. Missing values
# -----------------------------------------

print("Missing values:")
print(df.isnull().sum())

print()


# -----------------------------------------
# 6. Recovery distribution
# -----------------------------------------

print("Recovery distribution:")
print(df["recovered"].value_counts())

print()

print("Recovery percentage:")
print(df["recovered"].value_counts(normalize=True) * 100)

print()


# -----------------------------------------
# 7. Recovery by payment method
# -----------------------------------------

print("Recovery rate by payment method:")

payment_recovery = (
    df.groupby("payment_method")["recovered"]
    .mean()
    .sort_values(ascending=False)
)

print(payment_recovery)

print()


# -----------------------------------------
# 8. Recovery by failure reason
# -----------------------------------------

print("Recovery rate by failure reason:")

failure_recovery = (
    df.groupby("failure_reason")["recovered"]
    .mean()
    .sort_values(ascending=False)
)

print(failure_recovery)

print()


# -----------------------------------------
# 9. Recovery by retry count
# -----------------------------------------

print("Recovery rate by retry count:")

retry_recovery = (
    df.groupby("retry_count")["recovered"]
    .mean()
)

print(retry_recovery)

print()


# -----------------------------------------
# 10. Numerical statistics
# -----------------------------------------

print("Numerical statistics:")

print(df.describe())