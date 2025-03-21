import pandas as pd
from pathlib import Path
import shutil

def filter_and_rename_files(index_csv: str, target_csv: str, src_dir: str, dest_dir: str):
    """
    Filters and renames files based on the target CSV and copies them into a new folder structure.

    Parameters:
    - index_csv (str): Path to the index.csv file containing the existing file paths and document IDs.
    - target_csv (str): Path to the target CSV containing document IDs to filter and rename.
    - src_dir (str): The root directory where source files are located.
    - dest_dir (str): The destination directory where filtered files will be copied and renamed.
    """
    # Load the index and target CSVs
    index_df = pd.read_csv(index_csv)
    target_df = pd.read_csv(target_csv)

    # Merge both dataframes on the document ID
    merged_df = index_df.merge(target_df, left_on="document_id", right_on="source_id")  # Update column names as needed

    # Iterate through the merged list
    for _, row in merged_df.iterrows():
        # Extract relevant IDs and paths
        source_id = row["source_id"]  # From target CSV
        new_id = row["target_id"]     # From target CSV (new document ID)
        src_file_path = Path(row["src_path"])  # Original file path from index.csv
        dest_subfolder = Path(dest_dir) / str(new_id)  # Destination subfolder
        dest_file_path = dest_subfolder / f"{new_id}-joined.txt"  # Renamed destination file

        # Ensure the destination subfolder exists
        dest_subfolder.mkdir(parents=True, exist_ok=True)

        # Copy and rename the file
        try:
            shutil.copy(src_file_path, dest_file_path)
            print(f"Copied and renamed: {src_file_path} -> {dest_file_path}")
        except FileNotFoundError:
            print(f"Source file not found: {src_file_path}")
        except Exception as e:
            print(f"Error copying file {src_file_path}: {e}")

if __name__ == "__main__":
    # Paths to your files and directories
    INDEX_CSV = "path/to/index.csv"        # Replace with the path to your index.csv
    TARGET_CSV = "path/to/target.csv"      # Replace with the path to your target file
    SRC_DIR = "dat/greek/cleaned_parsed_data"  # Source directory
    DEST_DIR = "dat/greek/new_groups"         # Destination directory

    # Execute the function
    filter_and_rename_files(INDEX_CSV, TARGET_CSV, SRC_DIR, DEST_DIR)
