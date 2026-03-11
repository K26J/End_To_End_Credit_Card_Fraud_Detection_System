import pytest
from fastapi.testclient import TestClient
from app import app

# Create a fake web browser to talk to our API without actually starting the server
client = TestClient(app)

def test_health_check():
    """Test 1: Does the server wake up properly?"""
    response = client.get("/health_check")
    assert response.status_code == 200
    assert response.json() == {"status": "API is live and models are loaded."}

def test_predict_safe_transaction():
    """Test 2: Does the math work? Send a known safe transaction."""
    # This is the exact data we used yesterday!
    safe_payload = {
        "features": [
            -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388, 0.239599, 0.098698,
            0.363787, 0.090794, -0.551600, -0.617801, -0.991390, -0.311169, 1.468177, -0.470401,
            0.207971, 0.025791, 0.403993, 0.251412, -0.018307, 0.277838, -0.110474, 0.066928,
            0.128539, -0.189115, 0.133558, -0.021053, 149.62
        ]
    }
    
    # Send the fake POST request
    response = client.post("/predict", json=safe_payload)
    
    # Check the results
    assert response.status_code == 200
    data = response.json()
    
    # Assertions are the core of testing. If these are false, the test fails!
    assert "fraud_probability" in data
    assert data["is_fraud"] == False
    assert data["message"] == "Transaction approved."