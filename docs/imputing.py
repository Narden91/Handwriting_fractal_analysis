import os
import pandas as pd
import numpy as np
from pathlib import Path

def impute_missing_values(file_path):
    """
    Impute missing values in a CSV file with the most frequent value in each column.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    bool: True if successful, False otherwise
    """
    try:
        # Step 1: Load the CSV file
        print(f"Loading CSV file: {file_path}")
        df = pd.read_csv(file_path)
        
        # Step 2: Check for NaN values
        total_nan = df.isna().sum().sum()
        print(f"Total NaN values found: {total_nan}")
        
        if total_nan == 0:
            print("No NaN values found. No imputation needed.")
            return True
        
        # Print NaN values per column
        nan_counts = df.isna().sum()
        print("\nNaN values per column:")
        for column, count in nan_counts.items():
            if count > 0:
                print(f"  {column}: {count}")
        
        # Step 3: Impute NaN values with most frequent value
        print("\nImputing NaN values with most frequent value in each column...")
        
        # Iterate through columns
        for column in df.columns:
            missing_mask = df[column].isna()
            missing_count = missing_mask.sum()
            
            if missing_count > 0:
                # Handle numeric and categorical columns differently
                if pd.api.types.is_numeric_dtype(df[column]):
                    # For numeric columns, use most frequent value
                    most_frequent = df[column].mode()[0]
                    df.loc[missing_mask, column] = most_frequent
                    print(f"  Imputed {missing_count} NaN values in column '{column}' with {most_frequent}")
                else:
                    # For categorical/string columns, use most frequent value
                    most_frequent = df[column].mode()[0]
                    df.loc[missing_mask, column] = most_frequent
                    print(f"  Imputed {missing_count} NaN values in column '{column}' with '{most_frequent}'")
        
        # Step 4: Verify imputation
        remaining_nan = df.isna().sum().sum()
        print(f"\nRemaining NaN values after imputation: {remaining_nan}")
        
        # Step 5: Save back to the same file
        df.to_csv(file_path, index=False)
        print(f"Successfully saved imputed data back to {file_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return False

def process_csv_folder(folder_path):
    """
    Process all CSV files in a given folder
    
    Parameters:
    folder_path (str): Path to the folder containing CSV files
    
    Returns:
    tuple: (total_files_processed, successful_files, failed_files)
    """
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return 0, 0, 0
    
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return 0, 0, 0
    
    print(f"Found {len(csv_files)} CSV files in {folder_path}")
    
    # Process each CSV file
    successful_files = 0
    failed_files = 0
    
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        print(f"\n{'='*50}")
        print(f"Processing file: {csv_file}")
        
        if impute_missing_values(file_path):
            successful_files += 1
        else:
            failed_files += 1
    
    return len(csv_files), successful_files, failed_files

def main():
    """
    Main function to get folder path and process all CSV files
    """
    # Get the folder path from user input
    folder_path = r"C:\Users\Emanuele\Documents\Progetti Python\Handwriting_fractal_analysis\data\Feature_fractal"
    
    # Process all CSV files in the folder
    total_files, successful_files, failed_files = process_csv_folder(folder_path)
    
    # Print summary
    print(f"\n{'='*50}")
    print("Summary:")
    print(f"Total CSV files found: {total_files}")
    print(f"Successfully processed: {successful_files}")
    print(f"Failed to process: {failed_files}")
    
    if total_files > 0:
        success_rate = (successful_files / total_files) * 100
        print(f"Success rate: {success_rate:.2f}%")

if __name__ == "__main__":
    main()