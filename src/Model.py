import pandas as pd
import numpy as np
import mlflow
import tensorflow as tf
import mlflow.tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import os
import yaml # FIXED 1: Added the YAML importer

# Load the parameters
with open("params.yaml", "r") as f:
   all_params = yaml.safe_load(f)

# Define Experiment
EXPERIMENT_NAME = all_params["experiment_name"]
exp_params = all_params[EXPERIMENT_NAME]

# Define function to train the model
def train_model():
   # Load the data
   data_path = os.path.join("data", "train_data_scaled.csv") # FIXED 2: Fixed spelling of "scaled"
   df = pd.read_csv(data_path)

   # Separate the X and y
   X_train = df.iloc[:, :-1]
   y_train = df.iloc[:, -1]

   # Start MLFlow autolog which automatically logs all the important parameters
   mlflow.tensorflow.autolog()

   # Start the Experiment tracking
   with mlflow.start_run(run_name=EXPERIMENT_NAME):
      # Define the model architecture
      model = Sequential()
      # FIXED 3: Changed 'params' to 'exp_params'
      model.add(Dense(exp_params["layer_1_nuerons"], activation='relu', input_dim=exp_params["input_dim"]))
      model.add(Dense(exp_params["layer_2_nuerons"], activation='relu'))
      model.add(Dense(exp_params["layer_3_nuerons"], activation='relu'))
      model.add(Dense(exp_params["out_layer_nuerons"], activation='sigmoid'))

      # Compile the model
      # FIXED 4: Removed the underscore from 'binary_crossentropy'
      model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[tf.keras.metrics.Recall(), 'accuracy'])

      # Train the model with logic
      if EXPERIMENT_NAME == "baseline_model":
         print("Training the baseline model with no data balancing")
         model.fit(X_train, y_train, epochs=exp_params["epochs"], validation_split=exp_params["validation_split"])
      else:
         print(f"The model logic is not written yet {EXPERIMENT_NAME}")
         return
      
      # Save the models
      os.makedirs("models", exist_ok=True)
      model_path = os.path.join("models", "baseline_model.h5") # Added .h5 so Keras packs it into one file
      model.save(model_path)
      print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
   train_model()