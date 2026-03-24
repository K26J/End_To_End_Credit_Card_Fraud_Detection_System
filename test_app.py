import requests
import json

# Define the endpoint where your Docker container is listening
URL = "http://127.0.0.1:8000/predict"

def run_integration_test():
    print("--- Starting Docker Integration Test ---\n")
    
    # The exact Kaggle dataset row for a legitimate transaction ($149.62)
    payload = {
      "features": [
        -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388, 
        0.239599, 0.098698, 0.363787, 0.090794, -0.551600, -0.617801, 
        -0.991390, -0.311169, 1.468177, -0.470401, 0.207971, 0.025791, 
        0.403993, 0.251412, -0.018307, 0.277838, -0.110474, 0.066928, 
        0.128539, -0.189115, 0.133558, -0.021053, 149.62
      ]
    }

    try:
        # Fire the payload at the Docker container
        response = requests.post(URL, json=payload)
        
        if response.status_code == 200:
            print("Success! The container processed the transaction:")
            print(json.dumps(response.json(), indent=4))
        else:
            print(f"Failed. API returned Status Code: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("CRITICAL ERROR: Could not connect. Is the Docker container running?")

if __name__ == "__main__":
    run_integration_test()