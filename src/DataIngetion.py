import pandas as pd
import os

# Use relative paths! DVC runs this from the 'MLOps' root folder.
# '..' means go up one level to 'DataScienceProjects'
SOURCE_PATH = os.path.join("..", "Datasets", "creditcard.csv")
OUTPUT_FOLDER = "data"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "raw_data.csv")

def fetch_data(source_path, output_folder, output_file):
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # read the data
        data = pd.read_csv(source_path)

        # save the data
        data.to_csv(output_file, index=False)
        print("data fetched and saved successfully")

    except Exception as e:
        print(f" Error occured while fetching and saving the data: {e}")
        raise e

if __name__ == "__main__":
    fetch_data(SOURCE_PATH, OUTPUT_FOLDER, OUTPUT_FILE)
    print("Data ingestion completed successfully")