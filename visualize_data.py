import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------------
# 1. Load dataset
# -----------------------------------------

df = pd.read_csv("data/payments.csv")


# -----------------------------------------
# 2. Recovery distribution
# -----------------------------------------

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="recovered"
)

plt.title("Payment Recovery Distribution")
plt.xlabel("Recovered")
plt.ylabel("Number of Payments")

plt.show()


# -----------------------------------------
# 3. Recovery by failure reason
# -----------------------------------------

plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="failure_reason",
    y="recovered"
)

plt.title("Recovery Rate by Failure Reason")
plt.xlabel("Failure Reason")
plt.ylabel("Recovery Rate")

plt.xticks(rotation=30)

plt.show()


# -----------------------------------------
# 4. Recovery by retry count
# -----------------------------------------

plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="retry_count",
    y="recovered"
)

plt.title("Recovery Rate by Retry Count")
plt.xlabel("Retry Count")
plt.ylabel("Recovery Rate")

plt.show()


# -----------------------------------------
# 5. Recovery by payment method
# -----------------------------------------

plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="payment_method",
    y="recovered"
)

plt.title("Recovery Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Recovery Rate")

plt.show()


# -----------------------------------------
# 6. Previous success rate
# -----------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="recovered",
    y="previous_success_rate"
)

plt.title("Previous Success Rate vs Recovery")
plt.xlabel("Recovered")
plt.ylabel("Previous Success Rate")

plt.show()