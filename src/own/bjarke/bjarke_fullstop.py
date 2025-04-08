import os
import csv
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        filename="process_log.log", 
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def split_text(text):
    """Splits text at full stops and ensures each sentence ends with a period."""
    sentences = [s.strip() + "." if s and not s.endswith(".") else s.strip() 
                 for s in text.split(".") if s.strip()]
    return "\n".join(sentences)

def process_file(file_path):
    """Reads a joined text file, splits sentences, and writes the processed version."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        processed_text = split_text(text)
        
        output_file = file_path.with_name(f"{file_path.stem.replace('-joined', '')}-fullstop.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(processed_text)
        
        logging.info(f"Processed: {file_path} -> {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Failed to process {file_path}: {e}")
        return None

def process_directory(base_dir, metadata_csv):
    """Iterates through metadata CSV to find and process -joined.txt files."""
    processed_files = []
    
    with open(metadata_csv, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            joined_file = Path(base_dir) / row['destination_relative_path']
            
            if joined_file.exists() and "-joined.txt" in joined_file.name:
                output_path = process_file(joined_file)
                if output_path:
                    processed_files.append((joined_file, output_path))
    
    # Save metadata
    metadata_path = Path(base_dir) / "fullstop_metadata.csv"
    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original_file", "processed_file"])
        writer.writerows(processed_files)

def main():
    setup_logging()
    base_dir = "dat/export/bjarke_test"  # Modify if needed
    metadata_csv = "dat/export/bjarke_test/new_index.csv"
    
    process_directory(base_dir, metadata_csv)
    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
