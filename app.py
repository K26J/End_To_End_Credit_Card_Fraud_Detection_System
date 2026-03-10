from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Initialize the FastAPI app
app= FastAPI(
    title= "Credit Card Fraud Detection API",
    description= "An MLOps API that serves our champion model.",
    version= "1.0"
)

MODEL_PATH= os.path.join("models", "final_model.h5")
print(f"Loading the model from {MODEL_PATH}")
model= load_model(MODEL_PATH)


# 2. Load the Scaler
SCALER_PATH = os.path.join("models", "scaler.pkl")
print(f"Loading the scaler from {SCALER_PATH}")
scaler = joblib.load(SCALER_PATH)

# FIXED: Replaced the '=' with a ':' for Type Hinting
class Transaction(BaseModel):
    features: list[float]


@app.get("/")
def health_check():
    # FIXED: Typo "Satus" -> "Status"
    return {"Status": "API is live and the model is awake"}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    if len(transaction.features) != 29:
        raise HTTPException(status_code= 400, detail= "Invalid data: Must provide exactly 29 features.")
    
    # Convert the json in numpy array for keras
    # Reshape to (1, 29)
    data= np.array(transaction.features).reshape(1, -1)

    # Scaling
    raw_amount= data[0,28].reshape(1, -1)
    scaled_amount= scaler.transform(raw_amount)[0][0]

    # Replace the old raw amount with the new scaled amount in the array
    data[0, 28] = scaled_amount

    # Get the raw probability from the model
    prob= model.predict(data)[0][0]

    # Convert the mathematical threshold into a business decision
    is_fraud= bool(prob > 0.35)

    # Return the clean JSON response to user
    # FIXED: Typo "messege" -> "message"
    return {
        "fraud_probability": float(prob),
        "is_fraud": is_fraud,
        "message": "Transaction blocked!" if is_fraud else "Transaction approved."
    }