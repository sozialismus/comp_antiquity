import os
import csv
import pandas as pd
from pathlib import Path
from shutil import copy2
import logging
import re
import argparse

CLEANED_DATA_PATH = "/home/gnosis/Documents/au_work/main/comp_antiquity/dat/greek/bjarke_test/"

def update_document_id(document_id: str) -> str:
    """Switch between grc1 and grc2 in document_id."""
    if "grc1" in document_id:
        return re.sub(r'grc1', 'grc2', document_id)
    elif "grc2" in document_id:
        return re.sub(r'grc2', 'grc1', document_id)
    return document_id  # If neither, return as-is

def read_index(csv_path: str) -> dict:
    """
    Reads the CSV file and returns a dictionary mapping document_id to the file path,
    including alternate versions via update_document_id.
    """
    index = {}
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')  # Adjust delimiter if needed
        for row in reader:
            document_id = row['document_id']
            # Use 'src_path' column to get the file path
            src_path = row.get('src_path')
            if not src_path:
                continue  # Skip rows without a file path
            
            # Ensure src_path starts with CLEANED_DATA_PATH
            if not src_path.startswith(CLEANED_DATA_PATH):
                continue  # Skip unexpected paths

            # Store the original ID mapping
            index[document_id] = src_path

            # Also add the alternate document ID (switch grc1 <-> grc2) if not already present
            updated_document_id = update_document_id(document_id)
            if updated_document_id not in index:
                index[updated_document_id] = src_path

            print(f"Loaded {document_id} (also checking {updated_document_id}) -> {src_path}")
    return index

def export_target_files(src_dir: str, source_index_path: str, target_csv_path: str, dest_dir: str, log_file: str):
    """
    Exports a subsection of files identified in target_csv based on metadata in source_index.csv.
    Files are copied to dest_dir (preserving their source subfolder), renamed with the new document ID,
    and an index CSV is generated.
    """
    dest_dir_path = Path(dest_dir)
    dest_dir_path.mkdir(parents=True, exist_ok=True)
    
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(filename=str(log_file_path), level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        source_index_df = pd.read_csv(source_index_path)
        target_csv_df = pd.read_csv(target_csv_path)
    except Exception as e:
        logging.error(f"Error loading CSV files: {e}")
        return

    # Build index dictionary from source_index; using read_index from earlier
    index_dict = read_index(source_index_path)
    
    if target_csv_df.empty:
        logging.warning("Target CSV is empty; no files to process.")
        return

    export_index_data = []
    export_index_csv_path = None  # Initialize variable

    for _, row in target_csv_df.iterrows():
        try:
            original_doc_id = row["document_id"]
            new_document_id = str(row["updated_ids"]).strip()
            
            # Check for matching document_id in the index, or its alternate
            if original_doc_id in index_dict:
                matched_id = original_doc_id
            else:
                alt_id = update_document_id(original_doc_id)
                if alt_id in index_dict:
                    matched_id = alt_id
                    logging.info(f"Document ID updated: {original_doc_id} matched via alternate {alt_id}")
                else:
                    logging.error(f"Document ID {original_doc_id} (or its alternate) not found in index.")
                    continue

            src_path = index_dict[matched_id]
            full_src_path = Path(src_path)
            # Compute the relative subpath using the fixed base: CLEANED_DATA_PATH
            relative_subpath = full_src_path.relative_to(Path(CLEANED_DATA_PATH))
            src_subfolder = relative_subpath.parent
            
            # Build destination folder: dest_dir/<src_subfolder>/<new_document_id>/texts/
            new_file_dir = dest_dir_path / src_subfolder / str(new_document_id) / "texts"
            new_file_dir.mkdir(parents=True, exist_ok=True)
            
            # Destination file path: <new_document_id>-joined.txt in new_file_dir
            dest_file_path = new_file_dir / f"{new_document_id}-joined.txt"
            
            copy2(src_path, dest_file_path)
            logging.info(f"Copied: {src_path} -> {dest_file_path}")
            
            export_index_data.append({
                "orig_document_id": original_doc_id,
                "new_document_id": new_document_id,
                "source_relative_path": str(relative_subpath),
                "destination_relative_path": str(dest_file_path.relative_to(dest_dir_path))
            })
        except Exception as e:
            logging.error(f"Error processing file {row.get('document_id', 'Unknown')}: {e}")

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

    if export_index_csv_path:
        print(f"Export completed. New index saved to {export_index_csv_path}. Check log at {log_file_path}.")
    else:
        print(f"Export completed, but no new index was created. Check log at {log_file_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export selected texts based on document IDs from the master index.")
    parser.add_argument("--src_dir", required=True, help="Base directory for cleaned_parsed_data.")
    parser.add_argument("--source_index", required=True, help="Path to the source index CSV file.")
    parser.add_argument("--target_csv", required=True, help="Path to the target CSV file with document IDs to extract.")
    parser.add_argument("--dest_dir", required=True, help="Destination directory for the exported files.")
    parser.add_argument("--log_file", required=True, help="Path to the log file for errors and updates.")
    args = parser.parse_args()
    
    print("Starting export process...")
    export_target_files(args.src_dir, args.source_index, args.target_csv, args.dest_dir, args.log_file)
    print("Export process complete.")
