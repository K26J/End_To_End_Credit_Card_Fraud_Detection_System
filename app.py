from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import warnings

# Suppress the scikit-learn feature name warning to keep production logs clean
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Initialize the FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="An MLOps API that serves our champion model.",
    version="1.0"
)

# 1. Load the Model
MODEL_PATH = os.path.join("models", "final_model.h5")
print(f"Loading the model from {MODEL_PATH}")
model = load_model(MODEL_PATH)

# 2. Load the Scaler
SCALER_PATH = os.path.join("models", "scaler.pkl")
print(f"Loading the scaler from {SCALER_PATH}")
scaler = joblib.load(SCALER_PATH)

class Transaction(BaseModel):
    features: list[float]

@app.get("/")
def health_check():
    return {"Status": "API is live and the model is awake"}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    # Ensure exactly 29 features are passed 
    if len(transaction.features) != 29:
        raise HTTPException(status_code=400, detail="Invalid data: Must provide exactly 29 features.")
    
    # Convert the JSON list into a numpy array for Keras
    data = np.array(transaction.features).reshape(1, -1)

    # Scale the entire feature array at once
    data_scaled = scaler.transform(data)

    # Get the raw probability from the model using the fully scaled data
    prob = model.predict(data_scaled)[0][0]

    # Convert the mathematical threshold into a business decision
    is_fraud = bool(prob > 0.35)

    # Return the clean JSON response to user
    return {
        "fraud_probability": float(prob),
        "is_fraud": is_fraud,
        "message": "Transaction blocked!" if is_fraud else "Transaction approved."
    }