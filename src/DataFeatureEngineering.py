import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Defining the paths
PATHS_DICT= {"TRAIN_DATA_INPUT_PATH": os.path.join("data", "train_data.csv" ),
             "TEST_DATA_INPUT_PATH": os.path.join("data", "test_data.csv"),
             "OUTPUT_FOLDER": "data",
             "TRAIN_PROCESSED_PATH": os.path.join("data", "train_data_scaled.csv"),
             "TEST_PROCESSED_PATH": os.path.join("data", "test_data_scaled.csv"),
             
                        }

# Define Preprocessing and Feature Engineering function

def process(paths_dict):
    try:
        # Unpacking the paths from the dictionary
        train_data_input_path= paths_dict["TRAIN_DATA_INPUT_PATH"]
        test_data_input_path= paths_dict["TEST_DATA_INPUT_PATH"]
        output_folder= paths_dict["OUTPUT_FOLDER"]
        train_processed_path= paths_dict["TRAIN_PROCESSED_PATH"]
        test_processed_path= paths_dict["TEST_PROCESSED_PATH"]

        # Make folder if not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Read the data files
        train_data= pd.read_csv(train_data_input_path).drop('Time', axis=1)
        test_data= pd.read_csv(test_data_input_path).drop('Time', axis= 1)

        # Scale the data
        features_to_scale= ['Amount']
        scaler= StandardScaler()
        train_data[features_to_scale]= scaler.fit_transform(train_data[features_to_scale])
        test_data[features_to_scale]= scaler.transform(test_data[features_to_scale])

# ---> NEW: Save the Scaler Artifact <---
        os.makedirs("models", exist_ok=True)
        scaler_path = os.path.join("models", "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler successfully saved to {scaler_path}")


        # Save the Scaled data
        train_data.to_csv(train_processed_path, index= False)
        test_data.to_csv(test_processed_path, index= False)

        # Print the Success Messege
        print(f"The final train and test data saved successfully to {train_processed_path} and {test_processed_path}")

    except Exception as e:
        print("Error Occured During the Preprocessing and Scaling")
        raise e 
    
if __name__== "__main__":
    process(PATHS_DICT)
