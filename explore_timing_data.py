import pandas as pd


# Load timing dataset
df = pd.read_csv("data/recovery_timing.csv")


print("===== DATASET OVERVIEW =====")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())


# Overall recovery rate
print("\n===== OVERALL RECOVERY RATE =====")

print(
    df["recovered"].value_counts(normalize=True)
)


# Recovery rate by retry delay
print("\n===== RECOVERY RATE BY RETRY DELAY =====")

print(
    df.groupby("retry_delay_minutes")["recovered"]
    .mean()
    .sort_index()
)


# Recovery rate by failure reason
print("\n===== RECOVERY RATE BY FAILURE REASON =====")

print(
    df.groupby("failure_reason")["recovered"]
    .mean()
    .sort_values(ascending=False)
)


# Recovery rate by retry delay and failure reason
print("\n===== RECOVERY RATE BY FAILURE REASON + DELAY =====")

timing_table = (
    df.groupby(
        ["failure_reason", "retry_delay_minutes"]
    )["recovered"]
    .mean()
    .reset_index()
)

print(timing_table.to_string(index=False))