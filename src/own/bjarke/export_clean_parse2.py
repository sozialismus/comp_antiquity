import os
import pandas as pd
from pathlib import Path
from shutil import copy2
import logging

def export_target_files(src_dir: str, source_index_path: str, target_csv_path: str, dest_dir: str, log_file: str):
    """
    Exports a subsection of files identified in target_csv based on metadata in source_index.csv,
    copying them to new_groups while preserving the source subfolder (e.g., 'perseus' or 'first1k'),
    renaming them with the new document ID, and generating an index CSV.
    
    Parameters:
      - src_dir (str): Base directory for cleaned_parsed_data.
      - source_index_path (str): Path to the index.csv in cleaned_parsed_data.
      - target_csv_path (str): Path to the target.csv with document_id and new document IDs in column "AB".
      - dest_dir (str): Destination directory for the exported files.
      - log_file (str): Path to the error log file.
    """
    # Ensure dest_dir exists
    dest_dir_path = Path(dest_dir)
    dest_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Ensure the log file's parent directory exists
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load the CSV files
    try:
        source_index = pd.read_csv(source_index_path)
        target_csv = pd.read_csv(target_csv_path)
    except Exception as e:
        logging.error(f"Error loading CSV files: {e}")
        return

    # Merge on "document_id" (ensure both CSVs have the same key)
    merged_df = pd.merge(target_csv, source_index, on="document_id", how="inner")

    if merged_df.empty:
        logging.warning("No files were processed; merged DataFrame is empty.")
        return

    # Prepare metadata for the new index
    export_index_data = []

    # Iterate over each matched row
    for _, row in merged_df.iterrows():
        try:
            original_doc_id = row["document_id"]
            # New document ID from target CSV; adjust column name if needed.
            new_document_id = row["AB"]
            
            # The index CSV's "dest_path" is assumed to be a full relative path,
            # e.g. "dat/greek/cleaned_parsed_data/perseus/tlg0001.tlg001.perseus-grc2.txt".
            # Compute the relative path from src_dir.
            full_src_path = Path(row["dest_path"])
            relative_subpath = full_src_path.relative_to(src_dir)
            src_subfolder = relative_subpath.parent  # e.g. "perseus"
            
            # The actual source file path is built from src_dir and the relative_subpath.
            src_file_path = Path(src_dir) / relative_subpath

            # Build destination folder: dest_dir/<src_subfolder>/<new_document_id>
            new_file_dir = dest_dir_path / src_subfolder / str(new_document_id)
            new_file_dir.mkdir(parents=True, exist_ok=True)

            # Destination file path: <new_document_id>-joined.txt in new_file_dir
            dest_file_path = new_file_dir / f"{new_document_id}-joined.txt"

            # Copy the file from source to destination with the new name
            copy2(src_file_path, dest_file_path)
            logging.info(f"Copied: {src_file_path} -> {dest_file_path}")

            # Record metadata (store relative paths with respect to dest_dir)
            export_index_data.append({
                "orig_document_id": original_doc_id,
                "new_document_id": new_document_id,
                "source_relative_path": str(relative_subpath),
                "destination_relative_path": str(dest_file_path.relative_to(dest_dir_path))
            })
        except Exception as e:
            logging.error(f"Error processing file {row['document_id']}: {e}")

    # Save new metadata index CSV directly in dest_dir
    if export_index_data:
        export_index_df = pd.DataFrame(export_index_data)
        export_index_csv_path = dest_dir_path / "new_index.csv"
        try:
            export_index_df.to_csv(export_index_csv_path, index=False, encoding="utf-8")
            logging.info(f"New metadata index saved to {export_index_csv_path}")
        except Exception as e:
            logging.error(f"Error saving new index CSV: {e}")
    else:
        logging.warning("No files were processed; new metadata index not created.")

    print(f"Export completed. New index saved to {export_index_csv_path}. Errors logged to {log_file_path}.")


if __name__ == "__main__":
    # Define your file paths and directories
    SRC_DIR = "dat/greek/cleaned_parsed_data"  # Base directory for source files
    SOURCE_INDEX_PATH = f"{SRC_DIR}/index.csv"   # Source index CSV path
    TARGET_CSV_PATH = "dat/target.csv"           # Target CSV path
    DEST_DIR = "dat/export/new_groups"           # Destination directory; new_groups will be created here
    LOG_FILE = f"{DEST_DIR}/errors.log"          # Log file path

    print("Starting export process...")
    export_target_files(SRC_DIR, SOURCE_INDEX_PATH, TARGET_CSV_PATH, DEST_DIR, LOG_FILE)
    print("Export process complete.")
