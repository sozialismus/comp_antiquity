import pandas as pd
from pathlib import Path
import shutil
import logging

def filter_and_rename_files(index_csv: str, target_csv: str, src_dir: str, dest_dir: str):
    """
    Filters files from index_csv based on target_csv data, copies them from src_dir using
    the 'dest_path' from index.csv, and saves them in a new folder structure under dest_dir/new_groups.
    
    The target CSV must contain:
      - "document_id": which matches index.csv's "document_id"
      - "AB": the new document ID (to be used for renaming)
      
    The index CSV must contain at least:
      - "document_id"
      - "dest_path": the relative path (e.g. "perseus/tlg0001.tlg001.perseus-grc2.txt")
      
    Files will be copied such that:
      dest_dir/new_groups/<source_subfolder>/<new_id>/<new_id>-joined.txt
      
    Additionally, a new metadata index CSV is saved and errors are logged.
    """
    # Ensure the destination directory exists
    dest_dir_path = Path(dest_dir)
    dest_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file in the destination directory
    log_file = dest_dir_path / "export_log.log"
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load the CSV files
    try:
        index_df = pd.read_csv(index_csv)
        target_df = pd.read_csv(target_csv)
    except Exception as e:
        logging.error(f"Error loading CSV files: {e}")
        return

    # Merge both dataframes on the common "document_id" column
    try:
        merged_df = index_df.merge(target_df, on="document_id")
    except Exception as e:
        logging.error(f"Error merging CSV files: {e}")
        return

    # Prepare a list to collect metadata records for a new index file
    new_index_records = []

    # Iterate through each merged row
    for _, row in merged_df.iterrows():
        try:
            original_doc_id = row["document_id"]
            new_id = row["AB"]  # new document id from target CSV
            # Use 'dest_path' (a relative path like "perseus/filename.txt") to locate the file.
            src_file_path = Path(src_dir) / row["dest_path"]

            # Preserve the source substructure:
            # Get the relative directory of the file from dest_path (e.g., "perseus")
            src_relative_dir = Path(row["dest_path"]).parent
            # Build destination folder: dest_dir/new_groups/<src_relative_dir>/<new_id>
            dest_subfolder = dest_dir_path / "new_groups" / src_relative_dir / str(new_id)
            dest_subfolder.mkdir(parents=True, exist_ok=True)

            # Destination file: renamed to <new_id>-joined.txt within that folder.
            dest_file_path = dest_subfolder / f"{new_id}-joined.txt"

            # Copy the file from source to destination with the new name.
            shutil.copy(src_file_path, dest_file_path)
            logging.info(f"Copied: {src_file_path} -> {dest_file_path}")

            # Record metadata for this file (store relative paths with respect to dest_dir)
            record = {
                "original_document_id": original_doc_id,
                "new_document_id": new_id,
                "source_relative_path": str(Path(row["dest_path"])),
                "destination_relative_path": str(dest_file_path.relative_to(dest_dir_path))
            }
            new_index_records.append(record)
        except FileNotFoundError:
            error_msg = f"Source file not found: {src_file_path}"
            logging.error(error_msg)
        except Exception as e:
            error_msg = f"Error processing file {src_file_path}: {e}"
            logging.error(error_msg)

    # Save new metadata index CSV in the destination directory
    if new_index_records:
        new_index_df = pd.DataFrame(new_index_records)
        new_index_csv_path = dest_dir_path / "new_index.csv"
        try:
            new_index_df.to_csv(new_index_csv_path, index=False)
            logging.info(f"New metadata index saved to {new_index_csv_path}")
        except Exception as e:
            logging.error(f"Error saving new index CSV: {e}")
    else:
        logging.warning("No files were processed; new metadata index not created.")

if __name__ == "__main__":
    # Define your file paths and directories (adjust as needed)
    INDEX_CSV = "dat/greek/cleaned_parsed_data/index.csv"  # Path to index.csv containing 'document_id' and 'dest_path'
    TARGET_CSV = "dat/target.csv"            # Path to target.csv containing 'document_id' and 'AB'
    SRC_DIR = "dat/greek/cleaned_parsed_data"      # Base directory for source files (dest_path is relative to this)
    DEST_DIR = "dat/greek/export"                  # Base directory for export; new_groups will be created here

    filter_and_rename_files(INDEX_CSV, TARGET_CSV, SRC_DIR, DEST_DIR)

