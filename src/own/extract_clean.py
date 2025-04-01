import os
import csv
import shutil
import argparse
import re

CLEANED_DATA_PATH = "dat/greek/cleaned_parsed_data/"

def update_document_id(document_id):
    """Switch between grc1 and grc2 in document_id."""
    if "grc1" in document_id:
        return re.sub(r'grc1', 'grc2', document_id)
    elif "grc2" in document_id:
        return re.sub(r'grc2', 'grc1', document_id)
    return document_id  # If neither, return as-is

def read_index(csv_path):
    """Reads the CSV file and returns a dictionary mapping document_id to dest_path."""
    index = {}
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')  # Adjust delimiter if needed
        for row in reader:
            document_id = row['document_id']
            dest_path = row['dest_path']

            # Ensure dest_path starts with CLEANED_DATA_PATH
            if not dest_path.startswith(CLEANED_DATA_PATH):
                continue  # Skip unexpected paths

            # Store original ID
            index[document_id] = dest_path

            # Store possible alternate ID (grc1 <-> grc2)
            updated_document_id = update_document_id(document_id)
            if updated_document_id not in index:  # Avoid overwriting real entries
                index[updated_document_id] = dest_path

            print(f"Loaded {document_id} (also checking {updated_document_id}) -> {dest_path}")

    return index

def extract_files(index, selected_ids, output_dir):
    """Copies selected files while preserving the structure after CLEANED_DATA_PATH."""
    for document_id in selected_ids:
        # Check if the document_id exists, else try the updated grc1 <-> grc2 version
        if document_id in index:
            matched_id = document_id
        else:
            alt_id = update_document_id(document_id)
            matched_id = alt_id if alt_id in index else None

        if matched_id:
            src_path = index[matched_id]
            # Extract the relative path after "dat/greek/cleaned_parsed_data/"
            relative_path = src_path[len(CLEANED_DATA_PATH):]  
            dest_path = os.path.join(output_dir, relative_path)

            # Ensure destination folders exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Copy the file
            shutil.copy(src_path, dest_path)
            print(f"✅ Copied {src_path} -> {dest_path}")
        else:
            print(f"❌ document_id {document_id} (or its updated version) not found in index.")

def main():
    parser = argparse.ArgumentParser(description="Extract selected texts based on document_id.")
    parser.add_argument("csv_file", help="Path to the index CSV file.")
    parser.add_argument("output_dir", help="Directory where selected files will be copied.")
    parser.add_argument("document_ids", nargs='+', help="List of document_id values to extract.")
    args = parser.parse_args()
    
    index = read_index(args.csv_file)
    extract_files(index, args.document_ids, args.output_dir)
    
    print("🎉 Extraction complete!")

if __name__ == "__main__":
    main()
