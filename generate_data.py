import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Number of synthetic failed payments
NUMBER_OF_TRANSACTIONS = 10000


# --------------------------------------------------
# 1. Generate basic transaction information
# --------------------------------------------------

data = {

    "transaction_id": range(
        1,
        NUMBER_OF_TRANSACTIONS + 1
    ),

    "amount": np.random.lognormal(
        mean=8,
        sigma=1,
        size=NUMBER_OF_TRANSACTIONS
    ).round(2),

    "payment_method": np.random.choice(
        [
            "UPI",
            "Card",
            "NetBanking",
            "Wallet"
        ],
        NUMBER_OF_TRANSACTIONS,
        p=[
            0.50,
            0.30,
            0.15,
            0.05
        ]
    ),

    "failure_reason": np.random.choice(
        [
            "Insufficient Funds",
            "Bank Declined",
            "Network Error",
            "Technical Error",
            "Expired Card"
        ],
        NUMBER_OF_TRANSACTIONS,
        p=[
            0.25,
            0.25,
            0.20,
            0.20,
            0.10
        ]
    ),

    "retry_count": np.random.randint(
        0,
        5,
        NUMBER_OF_TRANSACTIONS
    ),

    "transaction_hour": np.random.randint(
        0,
        24,
        NUMBER_OF_TRANSACTIONS
    ),

    "previous_success_rate": np.random.uniform(
        0,
        1,
        NUMBER_OF_TRANSACTIONS
    ),

    "customer_lifetime_value": np.random.lognormal(
        mean=9,
        sigma=1,
        size=NUMBER_OF_TRANSACTIONS
    ).round(2)
}


# Convert dictionary into DataFrame
df = pd.DataFrame(data)


# --------------------------------------------------
# 2. Create synthetic recovery behavior
# --------------------------------------------------

recovery_score = (

    0.35

    + 0.45 * df["previous_success_rate"]

    - 0.08 * df["retry_count"]

    + 0.10 * (
        df["failure_reason"]
        == "Network Error"
    )

    + 0.08 * (
        df["failure_reason"]
        == "Technical Error"
    )

    - 0.10 * (
        df["failure_reason"]
        == "Insufficient Funds"
    )

    - 0.08 * (
        df["failure_reason"]
        == "Expired Card"
    )
)


# Keep probability between 5% and 95%

recovery_probability = np.clip(
    recovery_score,
    0.05,
    0.95
)


# --------------------------------------------------
# 3. Generate recovery outcome
# --------------------------------------------------

df["recovered"] = np.random.binomial(
    1,
    recovery_probability
)


# --------------------------------------------------
# 4. Save dataset
# --------------------------------------------------

df.to_csv(
    "data/payments.csv",
    index=False
)


# --------------------------------------------------
# 5. Display information
# --------------------------------------------------

print("===================================")
print("RecoverAI Dataset Created!")
print("===================================")

print()

print("Number of transactions:")
print(len(df))

print()

print("Dataset shape:")
print(df.shape)

print()

print("First 5 records:")
print(df.head())

print()

print("Recovery distribution:")
print(df["recovered"].value_counts())

print()

print("Recovery rate:")
print(
    df["recovered"].mean()
)