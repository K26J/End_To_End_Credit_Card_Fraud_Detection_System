import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Defining the input and output file paths


# Defining the paths dictionary to pass to the function
ALL_PATHS= {"INPUT_PATH": os.path.join("data", "raw_data.csv"),
            "OUTPUT_FOLDER": "data",
            "TRAIN_FILE": os.path.join("data", "train_data.csv"),
            "TEST_FILE": os.path.join("data", "test_data.csv")
           }

# Defining the data split function
def split_data(all_paths, test_size= 0.2, random_state=42):
    try:
        # Unpacking the paths from the dictionary
        input_path= all_paths["INPUT_PATH"]
        output_folder= all_paths["OUTPUT_FOLDER"]
        train_file= all_paths["TRAIN_FILE"]
        test_file= all_paths["TEST_FILE"]

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Read the raw data
        data= pd.read_csv(input_path)

        # Split the data into train and test sets
        train_data, test_data= train_test_split(data, test_size= test_size, random_state=random_state)

        # Save the train and test data
        train_data.to_csv(train_file, index= False)
        test_data.to_csv(test_file, index= False)
        print("Data Split completed Successfully!")

    except Exception as e:
        print(f" Error occured while splitting the data: {e}")
        raise e
    
if __name__== "__main__":
    split_data(ALL_PATHS)
    print("Data split completed successfully")

        
        

        
