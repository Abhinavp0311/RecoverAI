import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve


# -----------------------------------------
# 1. Load dataset
# -----------------------------------------

df = pd.read_csv("data/payments.csv")


# -----------------------------------------
# 2. Separate features and target
# -----------------------------------------

X = df.drop(
    columns=["recovered", "transaction_id"]
)

y = df["recovered"]


# -----------------------------------------
# 3. Train/test split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------------
# 4. Define feature types
# -----------------------------------------

categorical_features = [
    "payment_method",
    "failure_reason"
]

numerical_features = [
    "amount",
    "retry_count",
    "transaction_hour",
    "previous_success_rate",
    "customer_lifetime_value"
]


# -----------------------------------------
# 5. Create preprocessor
# -----------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# -----------------------------------------
# 6. Process data
# -----------------------------------------

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


# -----------------------------------------
# 7. Train model
# -----------------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_processed,
    y_train
)


# -----------------------------------------
# 8. Get predicted probabilities
# -----------------------------------------

probabilities = model.predict_proba(
    X_test_processed
)[:, 1]


# -----------------------------------------
# 9. Calculate calibration values
# -----------------------------------------

fraction_of_positives, mean_predicted_value = (
    calibration_curve(
        y_test,
        probabilities,
        n_bins=10,
        strategy="uniform"
    )
)


# -----------------------------------------
# 10. Plot calibration curve
# -----------------------------------------

plt.figure(figsize=(7, 6))

plt.plot(
    mean_predicted_value,
    fraction_of_positives,
    marker="o",
    label="RecoverAI"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel("Mean predicted probability")
plt.ylabel("Actual recovery rate")

plt.title("RecoverAI Probability Calibration")

plt.legend()

plt.grid()

plt.show()