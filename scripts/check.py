#!/usr/bin/env python3
import os
import pandas as pd

def main():
    folder_path = "bac_preds"  # Folder containing your bacteria CSV files
    # Define a threshold: if a file has fewer than this many rows, print its values.
    # Adjust the threshold based on your expectations.
    row_threshold = 10

    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    csv_files = [fn for fn in os.listdir(folder_path) if fn.endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return

    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        try:
            # Read CSV using default settings (assumes a header row exists)
            df = pd.read_csv(file_path)
            print(f"File: {csv_file} | Shape: {df.shape}")

            # Check if the number of rows is below our expected threshold
            if df.shape[0] < row_threshold:
                print("Contents:")
                print(df)
            print("-" * 40)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

if __name__ == "__main__":
    main()

