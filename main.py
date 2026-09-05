from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import os


# Load trained model
pipeline = joblib.load(
    "models/recovery_pipeline_v2.pkl"
)


# Create FastAPI application
app = FastAPI(
    title="RecoverAI API",
    description="AI-powered revenue recovery and smart retry system",
    version="1.0"
)


# --------------------------------------------------
# Input format
# --------------------------------------------------

class PaymentRequest(BaseModel):
    customer_success_rate: float
    customer_lifetime_value: float
    amount: float
    payment_method: str
    failure_reason: str
    retry_count: int
    transaction_hour: int


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def home():
    return FileResponse("app/index.html")

@app.get("/dashboard")
def dashboard():

    df = pd.read_csv("data/payments_v2.csv")

    total_failed_value = df["amount"].sum()

    recovery_rate = df["recovered"].mean()

    X = df.drop(
        columns=[
            "recovered",
            "transaction_id",
            "customer_id"
        ]
    )

    probabilities = pipeline.predict_proba(X)[:, 1]

    expected_revenue = (
        df["amount"] * probabilities
    ).sum()

    return {
        "total_payments": len(df),
        "total_failed_value": round(
            float(total_failed_value), 2
        ),
        "recovery_rate": round(
            float(recovery_rate), 4
        ),
        "expected_recoverable_revenue": round(
            float(expected_revenue), 2
        )
    }
# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
# --------------------------------------------------
# Recovery opportunities endpoint
# --------------------------------------------------

@app.get("/opportunities")
def opportunities():

    df = pd.read_csv("data/payments_v2.csv")

    X = df.drop(
        columns=[
            "recovered",
            "transaction_id",
            "customer_id"
        ]
    )

    probabilities = pipeline.predict_proba(X)[:, 1]

    df["recovery_probability"] = probabilities

    df["expected_revenue"] = (
        df["amount"] *
        df["recovery_probability"]
    )

    df["priority_score"] = (
        df["expected_revenue"] *
        (1 - df["retry_count"] / 5)
    )

    def get_action(row):

        probability = row["recovery_probability"]
        failure_reason = row["failure_reason"]
        retry_count = row["retry_count"]

        if retry_count >= 3:
            return "DO_NOT_RETRY"

        if failure_reason == "Expired Card":
            return "UPDATE_PAYMENT_METHOD"

        if failure_reason in [
            "Network Error",
            "Technical Error"
        ]:
            if probability >= 0.50:
                return "RETRY"
            else:
                return "DO_NOT_RETRY"

        if failure_reason == "Insufficient Funds":
            return "WAIT_AND_NOTIFY"

        if failure_reason == "Bank Declined":
            if probability >= 0.60:
                return "RETRY_LATER"
            else:
                return "DO_NOT_RETRY"

        if probability >= 0.70:
            return "RETRY"

        elif probability >= 0.40:
            return "RETRY_LATER"

        else:
            return "DO_NOT_RETRY"

    df["recommended_action"] = df.apply(
        get_action,
        axis=1
    )

    timing = {
        "Network Error": 15,
        "Technical Error": 60,
        "Insufficient Funds": 1440,
        "Bank Declined": 360,
        "Expired Card": 360
    }

    df["retry_delay_minutes"] = (
        df["failure_reason"].map(timing)
    )

    df.loc[
        df["recommended_action"].isin([
            "DO_NOT_RETRY",
            "UPDATE_PAYMENT_METHOD",
            "WAIT_AND_NOTIFY"
        ]),
        "retry_delay_minutes"
    ] = None

    df = df.sort_values(
        "priority_score",
        ascending=False
    )

    result = df.head(20)[[
        "transaction_id",
        "amount",
        "failure_reason",
        "retry_count",
        "recovery_probability",
        "expected_revenue",
        "recommended_action",
        "retry_delay_minutes"
    ]]

    records = result.to_dict(
        orient="records"
    )

    for record in records:

        if pd.isna(record["retry_delay_minutes"]):
            record["retry_delay_minutes"] = None

    return records
# --------------------------------------------------
# Action distribution endpoint
# --------------------------------------------------

@app.get("/action-distribution")
def action_distribution():

    df = pd.read_csv("data/payments_v2.csv")

    X = df.drop(
        columns=[
            "recovered",
            "transaction_id",
            "customer_id"
        ]
    )

    probabilities = pipeline.predict_proba(X)[:, 1]

    df["recovery_probability"] = probabilities

    def get_action(row):

        probability = row["recovery_probability"]
        failure_reason = row["failure_reason"]
        retry_count = row["retry_count"]

        if retry_count >= 3:
            return "DO_NOT_RETRY"

        if failure_reason == "Expired Card":
            return "UPDATE_PAYMENT_METHOD"

        if failure_reason in [
            "Network Error",
            "Technical Error"
        ]:
            if probability >= 0.50:
                return "RETRY"
            else:
                return "DO_NOT_RETRY"

        if failure_reason == "Insufficient Funds":
            return "WAIT_AND_NOTIFY"

        if failure_reason == "Bank Declined":
            if probability >= 0.60:
                return "RETRY_LATER"
            else:
                return "DO_NOT_RETRY"

        if probability >= 0.70:
            return "RETRY"

        elif probability >= 0.40:
            return "RETRY_LATER"

        else:
            return "DO_NOT_RETRY"

    df["recommended_action"] = df.apply(
        get_action,
        axis=1
    )

    distribution = (
        df["recommended_action"]
        .value_counts()
        .to_dict()
    )

    return distribution   

@app.post("/predict")
def predict(payment: PaymentRequest):

    data = pd.DataFrame([{
        "customer_success_rate": payment.customer_success_rate,
        "customer_lifetime_value": payment.customer_lifetime_value,
        "amount": payment.amount,
        "payment_method": payment.payment_method,
        "failure_reason": payment.failure_reason,
        "retry_count": payment.retry_count,
        "transaction_hour": payment.transaction_hour
    }])

    probability = pipeline.predict_proba(data)[0][1]

    expected_revenue = (
        payment.amount * probability
    )

    # Recovery action
    if payment.retry_count >= 3:
        action = "DO_NOT_RETRY"

    elif payment.failure_reason == "Expired Card":
        action = "UPDATE_PAYMENT_METHOD"

    elif payment.failure_reason in [
        "Network Error",
        "Technical Error"
    ]:
        if probability >= 0.50:
            action = "RETRY"
        else:
            action = "DO_NOT_RETRY"

    elif payment.failure_reason == "Insufficient Funds":
        action = "WAIT_AND_NOTIFY"

    elif payment.failure_reason == "Bank Declined":
        if probability >= 0.60:
            action = "RETRY_LATER"
        else:
            action = "DO_NOT_RETRY"

    elif probability >= 0.70:
        action = "RETRY"

    elif probability >= 0.40:
        action = "RETRY_LATER"

    else:
        action = "DO_NOT_RETRY"

    # Smart retry timing
    timing = {
        "Network Error": 15,
        "Technical Error": 60,
        "Insufficient Funds": 1440,
        "Bank Declined": 360,
        "Expired Card": 360
    }

    retry_delay = timing.get(
        payment.failure_reason
    )

    # No timing for actions that don't retry
    if action in [
        "DO_NOT_RETRY",
        "UPDATE_PAYMENT_METHOD",
        "WAIT_AND_NOTIFY"
    ]:
        retry_delay = None

    return {
        "recovery_probability": round(
            float(probability), 4
        ),
        "expected_revenue": round(
            float(expected_revenue), 2
        ),
        "recommended_action": action,
        "recommended_retry_delay_minutes": retry_delay
    }