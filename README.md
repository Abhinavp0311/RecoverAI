# RecoverAI — AI-Powered Revenue Recovery & Smart Retry System

RecoverAI is a machine learning and decision-support system designed to predict whether a failed digital payment can be recovered and recommend the most appropriate recovery action.
RecoverAI combines machine learning predictions with business decision rules to estimate recoverable revenue, recommend recovery actions, and suggest retry timing for failed digital payments.

> **Note:** This project uses fully synthetic payment data for demonstration and portfolio purposes. It does not use Razorpay or any proprietary/internal payment data.
## 🚀 Project Overview

Failed payments represent potential lost revenue for digital businesses.

RecoverAI attempts to answer four practical questions:

1. **Can this failed payment be recovered?**
2. **How much revenue is potentially recoverable?**
3. **What action should be taken?**
4. **When should the payment be retried?**
## 🧠 How RecoverAI Works

```text
Failed Payment
      │
      ▼
Customer + Transaction Features
      │
      ▼
ML Recovery Prediction
      │
      ├── Recovery Probability
      │
      ▼
Expected Recoverable Revenue
      │
      ▼
Recovery Decision Engine
      │
      ├── RETRY
      ├── WAIT_AND_NOTIFY
      ├── UPDATE_PAYMENT_METHOD
      └── DO_NOT_RETRY
      │
      ▼
Retry Timing Recommendation
      │
      ▼
Recovery Opportunities Dashboard

## 📊 Machine Learning Model

The current RecoverAI model uses **Logistic Regression** with a preprocessing pipeline.

### Features

**Numerical features:**

- Customer success rate
- Customer lifetime value
- Payment amount
- Retry count
- Transaction hour

**Categorical features:**

- Payment method
- Failure reason

The preprocessing pipeline uses:

- StandardScaler for numerical features
- OneHotEncoder for categorical features
- Logistic Regression for recovery probability prediction

## 📈 Model Performance

The current transaction-level evaluation produced:

| Metric | Score |
|---|---:|
| Accuracy | 68.10% |
| Precision | 56.81% |
| Recall | 39.11% |
| F1 Score | 46.33% |
| ROC-AUC | 69.61% |

The model is primarily used to **rank and prioritize recovery opportunities** rather than treating one probability threshold as an absolute business decision.

## 💰 Expected Recoverable Revenue

For each failed payment, RecoverAI estimates potential recoverable revenue using:

```text
Expected Revenue =
Payment Amount × Recovery Probability

## ⚙️ Recovery Decision Engine

RecoverAI converts model predictions into practical recovery actions.

### RETRY

Recommended when the payment has a favorable recovery probability and another attempt is appropriate.

### WAIT_AND_NOTIFY

Used for situations such as insufficient funds where immediately retrying may not be useful.

### UPDATE_PAYMENT_METHOD

Used for payment failures such as expired cards where changing the payment method is more appropriate than retrying.

### DO_NOT_RETRY

Used when repeated retries or low recovery probability make another attempt unlikely to be worthwhile.

## ⏱️ Smart Retry Timing

RecoverAI includes a synthetic recovery-timing component that estimates useful retry delays by failure reason.

| Failure Reason | Recommended Delay |
|---|---:|
| Network Error | 15 minutes |
| Technical Error | 60 minutes |
| Bank Declined | 360 minutes |
| Expired Card | 360 minutes |
| Insufficient Funds | 1440 minutes |

These timings are **synthetic empirical recommendations** and should not be treated as production payment-network rules.

## 📊 Dashboard

The RecoverAI dashboard provides:

- **Estimated Recoverable Revenue**
- **Total Failed Value**
- **Recovery Rate**
- **Failed Payment Count**
- AI-powered payment analysis
- Recovery action distribution
- Prioritized recovery opportunities
- Filters by action and failure reason
- Recommended retry timing

## 🛠️ Tech Stack

- **Python**
- **pandas**
- **NumPy**
- **scikit-learn**
- **XGBoost**
- **FastAPI**
- **Uvicorn**
- **Pydantic**
- **Matplotlib**
- **Seaborn**
- **joblib**
- **HTML / CSS / JavaScript**

## 📁 Project Structure

```text
RecoverAI/
│
├── app/
│   ├── main.py
│   └── index.html
│
├── data/
├── models/
├── notebooks/
│
├── src/
│   ├── generate_data.py
│   ├── explore_data.py
│   ├── visualize_data.py
│   ├── train_model.py
│   ├── recovery_predictions.py
│   ├── calibration.py
│   ├── recovery_actions.py
│   ├── generate_timing_data.py
│   ├── explore_timing_data.py
│   ├── train_timing_model.py
│   ├── recommend_timing.py
│   ├── decision_engine.py
│   ├── generate_realistic_data.py
│   ├── explore_realistic_data.py
│   ├── evaluate_model.py
│   ├── evaluate_v2_grouped.py
│   └── optimize_threshold.py
│
├── .gitignore
├── README.md
└── ...

## ▶️ Running the Project

### 1. Create a virtual environment

Run this command in Command Prompt:

`python -m venv venv`

### 2. Activate the environment

Run this command in Command Prompt:

`venv\Scripts\activate.bat`

### 3. Install dependencies

Run this command in Command Prompt:

`pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter joblib fastapi uvicorn`

### 4. Start the API

Run this command in Command Prompt:

`uvicorn app.main:app --reload`

### 5. Open the dashboard

Open this in your browser:

http://127.0.0.1:8000/

API documentation:

http://127.0.0.1:8000/docs

## 🔬 Dataset

The primary dataset contains 15,000 synthetic failed-payment transactions across 3,000 synthetic customers.

The data includes:

- Customer success history
- Customer lifetime value
- Payment amount
- Payment method
- Failure reason
- Retry count
- Transaction hour
- Recovery outcome

## ⚠️ Limitations

- The dataset is synthetic and does not represent real payment-network behavior.
- The model predicts recovery likelihood but does not establish causality.
- Retry timings are synthetic empirical recommendations.
- Expected recoverable revenue is a probabilistic estimate, not guaranteed revenue.
- The model has not been validated against real-world payment data.

## 🔮 Future Improvements

- Cost-sensitive decision optimization
- Customer-level temporal validation
- Better probability calibration
- SHAP-based explainability
- More sophisticated retry policies
- A/B testing of recovery strategies
- Real-time event processing
- Model drift monitoring
- Production database integration
- Automated recovery workflows

## 🎯 Project Goal

RecoverAI demonstrates how machine learning can be combined with business rules and operational decision-making to create a practical revenue-recovery system.

Rather than simply predicting whether a payment will recover, the project focuses on turning predictions into actionable recovery decisions.

## 📌 Disclaimer

RecoverAI is an independent portfolio project.

All payment data used in this project is synthetic and created for educational and demonstration purposes.

The project does not use confidential, proprietary, or internal Razorpay data.