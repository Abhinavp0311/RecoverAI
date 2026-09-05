import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)
from sklearn.calibration import calibration_curve


# ==========================================
# 1. Load data and trained pipeline
# ==========================================

df = pd.read_csv("data/payments.csv")

pipeline = joblib.load(
    "models/recovery_pipeline.pkl"
)


# ==========================================
# 2. Prepare features
# ==========================================

X = df.drop(
    columns=["recovered", "transaction_id"]
)

y = df["recovered"]


# ==========================================
# 3. Create test set
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Make predictions
# ==========================================

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# ==========================================
# 5. Print performance metrics
# ==========================================

print("\n===== RECOVERAI MODEL EVALUATION =====\n")

print(
    "Accuracy :",
    round(accuracy_score(y_test, y_pred), 4)
)

print(
    "Precision:",
    round(precision_score(y_test, y_pred), 4)
)

print(
    "Recall   :",
    round(recall_score(y_test, y_pred), 4)
)

print(
    "F1 Score :",
    round(f1_score(y_test, y_pred), 4)
)

print(
    "ROC-AUC  :",
    round(roc_auc_score(y_test, y_probability), 4)
)


# ==========================================
# 6. Confusion matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n===== CONFUSION MATRIX =====\n")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Not Recovered",
        "Recovered"
    ]
)

disp.plot()

plt.title(
    "RecoverAI Confusion Matrix"
)

plt.tight_layout()

plt.show()


# ==========================================
# 7. ROC curve
# ==========================================

RocCurveDisplay.from_predictions(
    y_test,
    y_probability
)

plt.title(
    "RecoverAI ROC Curve"
)

plt.tight_layout()

plt.show()


# ==========================================
# 8. Calibration curve
# ==========================================

prob_true, prob_pred = calibration_curve(
    y_test,
    y_probability,
    n_bins=10,
    strategy="uniform"
)

plt.plot(
    prob_pred,
    prob_true,
    marker="o",
    label="RecoverAI"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel(
    "Predicted Probability"
)

plt.ylabel(
    "Actual Recovery Rate"
)

plt.title(
    "RecoverAI Probability Calibration"
)

plt.legend()

plt.tight_layout()

plt.show()


# ==========================================
# 9. Feature importance
# ==========================================

preprocessor = pipeline.named_steps[
    "preprocessor"
]

model = pipeline.named_steps[
    "model"
]


feature_names = (
    preprocessor
    .get_feature_names_out()
)

coefficients = model.coef_[0]


feature_importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
})


feature_importance["absolute_coefficient"] = (
    feature_importance["coefficient"]
    .abs()
)


feature_importance = (
    feature_importance
    .sort_values(
        "absolute_coefficient",
        ascending=False
    )
)


print(
    "\n===== TOP MODEL FEATURES =====\n"
)

print(
    feature_importance[
        [
            "feature",
            "coefficient"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ==========================================
# 10. Plot feature importance
# ==========================================

top_features = (
    feature_importance
    .head(10)
    .sort_values("coefficient")
)


plt.barh(
    top_features["feature"],
    top_features["coefficient"]
)

plt.xlabel(
    "Logistic Regression Coefficient"
)

plt.title(
    "RecoverAI Top Feature Effects"
)

plt.tight_layout()

plt.show()