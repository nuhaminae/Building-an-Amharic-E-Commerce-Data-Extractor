import os
import pandas as pd

def combine_csvs (csv_dir, data_dir):
    """
    Gets directory for CSV files and destination path and returns a concatnated Dataframe 
    of all the CSV files found in the directory.
    Excludes CSV that starts with 'telegram_data'.     

    Args:
    csv_dir (str): The path to CSV files
    data_dir (str): The path to the save the combiend CSV files.

    Returns:
        combined_df (pd.DataFrame): A concatnated DataFrame.    
    """
    
    # Filter to only CSVs scraped (exclude telegram_data.csv or misc)
    csv_files = [
        f for f in os.listdir(csv_dir) if f.endswith('.csv') and not f.startswith('telegram_data')
        ]

    # Read and concatenate all CSVs
    df_list = []
    for file in csv_files:
        path = os.path.join(csv_dir, file)
        df = pd.read_csv(path)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)

    # Save final result
    output_path = os.path.join(data_dir, 'telegram_data.csv')
    relative_path = os.path.relpath(output_path, os.getcwd())
    combined_df.to_csv(output_path, index=False)

    print(f"✓ Combined {len(csv_files)} CSVs into {relative_path}")