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
import os
import yaml 
from tensorflow.keras import regularizers

# Load the parameters
with open("params.yaml", "r") as f:
    all_params = yaml.safe_load(f)

EXPERIMENT_NAME = all_params["experiment_name"]
exp_params = all_params[EXPERIMENT_NAME]

def train_model():
    # Load the data
    data_path = os.path.join("data", "train_data_scaled.csv") 
    df = pd.read_csv(data_path)

    # Separate the X and y
    X_train = df.iloc[:, :-1]
    y_train = df.iloc[:, -1]

    # Start MLFlow autolog
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name=EXPERIMENT_NAME):
       
       if EXPERIMENT_NAME == "baseline_model":
          print("Training the baseline model with no data balancing")
          model = Sequential()
          model.add(Dense(exp_params["layer_1_neurons"], activation='relu', input_dim=exp_params["input_dim"]))
          model.add(Dense(exp_params["layer_2_neurons"], activation='relu'))
          model.add(Dense(exp_params["layer_3_neurons"], activation='relu'))
          model.add(Dense(exp_params["out_layer_neurons"], activation='sigmoid'))

          model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[tf.keras.metrics.Recall(name='recall'), 'accuracy'])
          model.fit(X_train, y_train, epochs=exp_params["epochs"], validation_split=exp_params["validation_split"])

       elif EXPERIMENT_NAME == "generalized_base_model":
          print(f"Training {EXPERIMENT_NAME} with Early Stopping (No Dropout)...")
          model = Sequential()
          
          model.add(Dense(exp_params["layer_1_neurons"], activation='relu', input_dim=exp_params["input_dim"]))
          model.add(Dense(exp_params["layer_2_neurons"], activation='relu'))
          model.add(Dense(exp_params["layer_3_neurons"], activation='relu'))
          model.add(Dense(exp_params["out_layer_neurons"], activation='sigmoid'))

          model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[tf.keras.metrics.Recall(name='recall'), 'accuracy'])

          es_params = exp_params["early_stopping"]
          call = EarlyStopping(
              monitor=es_params['monitor'],
              min_delta=es_params['min_delta'],
              patience=es_params['patience'],
              verbose=1,
              mode=es_params['mode'],
              restore_best_weights=es_params['restore_best_weights'],
              start_from_epoch=es_params['start_from_epoch']
          )
          
          model.fit(X_train, y_train, epochs=exp_params["epochs"], validation_split=exp_params["validation_split"], callbacks=[call])

          # ... [Previous baseline and generalized models up here] ...

       elif EXPERIMENT_NAME == "class_weights_model":
          print(f"Training {EXPERIMENT_NAME} with Class Weights and Early Stopping...")
          model = Sequential()
          
          model.add(Dense(exp_params["layer_1_neurons"], activation='relu', input_dim=exp_params["input_dim"]))
          model.add(Dense(exp_params["layer_2_neurons"], activation='relu'))
          model.add(Dense(exp_params["layer_3_neurons"], activation='relu'))
          model.add(Dense(exp_params["out_layer_neurons"], activation='sigmoid'))

          model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[tf.keras.metrics.Recall(name='recall'), 'accuracy'])

          es_params = exp_params["early_stopping"]
          call = EarlyStopping(
              monitor=es_params['monitor'],
              min_delta=es_params['min_delta'],
              patience=es_params['patience'],
              verbose=1,
              mode=es_params['mode'],
              restore_best_weights=es_params['restore_best_weights'],
              start_from_epoch=es_params['start_from_epoch']
          )
          
          # Notice the new class_weight argument added to model.fit!
          model.fit(
              X_train, 
              y_train, 
              epochs=exp_params["epochs"], 
              validation_split=exp_params["validation_split"], 
              callbacks=[call],
              class_weight=exp_params["class_weights"] 
          )

       elif EXPERIMENT_NAME == "custom_class_weights_model":
          print(f"Training {EXPERIMENT_NAME} with Class Weights and Early Stopping...")
          model = Sequential()
          
          model.add(Dense(exp_params["layer_1_neurons"], activation='relu', input_dim=exp_params["input_dim"]))
          model.add(Dense(exp_params["layer_2_neurons"], activation='relu'))
          model.add(Dense(exp_params["layer_3_neurons"], activation='relu'))
          model.add(Dense(exp_params["out_layer_neurons"], activation='sigmoid'))

          model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=[tf.keras.metrics.Recall(name='recall'), 'accuracy'])

          es_params = exp_params["early_stopping"]
          call = EarlyStopping(
              monitor=es_params['monitor'],
              min_delta=es_params['min_delta'],
              patience=es_params['patience'],
              verbose=1,
              mode=es_params['mode'],
              restore_best_weights=es_params['restore_best_weights'],
              start_from_epoch=es_params['start_from_epoch']
          )
          
          # Notice the new class_weight argument added to model.fit!
          model.fit(
              X_train, 
              y_train, 
              epochs=exp_params["epochs"], 
              validation_split=exp_params["validation_split"], 
              callbacks=[call],
              class_weight=exp_params["class_weights"] 
          )

       elif EXPERIMENT_NAME == "controlled_class_weights_model":
          print(f"Training {EXPERIMENT_NAME} with Class Weights and Early Stopping...")
          model = Sequential()
          
          model.add(Dense(exp_params["layer_1_neurons"], activation='relu', input_dim=exp_params["input_dim"]))
          model.add(Dense(exp_params["layer_2_neurons"], activation='relu', kernel_regularizer= regularizers.l2(exp_params["l2_layer_2"])))
          model.add(Dropout(exp_params["layer_2_dropout"]))
          model.add(Dense(exp_params["layer_3_neurons"], activation='relu', kernel_regularizer= regularizers.l2(exp_params["l2_layer_3"])))
          model.add(Dropout(exp_params["layer_3_dropout"]))
          model.add(Dense(exp_params["out_layer_neurons"], activation='sigmoid'))

          optimizer= Adam(learning_rate= exp_params["learning_rate"])

          model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=[tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(curve='PR', name='pr_auc')])

          es_params = exp_params["early_stopping"]
          call = EarlyStopping(
              monitor=es_params['monitor'],
              min_delta=es_params['min_delta'],
              patience=es_params['patience'],
              verbose=1,
              mode=es_params['mode'],
              restore_best_weights=es_params['restore_best_weights'],
              start_from_epoch=es_params['start_from_epoch']
          )
          
          # Notice the new class_weight argument added to model.fit!
          model.fit(
              X_train, 
              y_train, 
              epochs=exp_params["epochs"], 
              validation_split=exp_params["validation_split"], 
              callbacks=[call],
              class_weight=exp_params["class_weights"] 
          )

       elif EXPERIMENT_NAME == "final_model":
          print(f"Training {EXPERIMENT_NAME} with Class Weights and Early Stopping...")
          model = Sequential()
          
          model.add(Dense(exp_params["layer_1_neurons"], activation='relu', input_dim=exp_params["input_dim"]))
          model.add(Dense(exp_params["layer_2_neurons"], activation='relu', kernel_regularizer= regularizers.l2(exp_params["l2_layer_2"])))
          model.add(Dropout(exp_params["layer_2_dropout"]))
          model.add(Dense(exp_params["layer_3_neurons"], activation='relu', kernel_regularizer= regularizers.l2(exp_params["l2_layer_3"])))
          model.add(Dropout(exp_params["layer_3_dropout"]))
          model.add(Dense(exp_params["out_layer_neurons"], activation='sigmoid'))

          optimizer= Adam(learning_rate= exp_params["learning_rate"])

          model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=[tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(curve='PR', name='pr_auc')])

          es_params = exp_params["early_stopping"]
          call = EarlyStopping(
              monitor=es_params['monitor'],
              min_delta=es_params['min_delta'],
              patience=es_params['patience'],
              verbose=1,
              mode=es_params['mode'],
              restore_best_weights=es_params['restore_best_weights'],
              start_from_epoch=es_params['start_from_epoch']
          )
          
          # Notice the new class_weight argument added to model.fit!
          model.fit(
              X_train, 
              y_train, 
              epochs=exp_params["epochs"], 
              validation_split=exp_params["validation_split"], 
              callbacks=[call],
              class_weight=exp_params["class_weights"] 
          )

       else:
          print(f"The model logic is not written yet: {EXPERIMENT_NAME}")
          return
       
       # Dynamic Saving
       os.makedirs("models", exist_ok=True)
       model_path = os.path.join("models", f"{EXPERIMENT_NAME}.h5") 
       model.save(model_path)
       print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
   train_model()