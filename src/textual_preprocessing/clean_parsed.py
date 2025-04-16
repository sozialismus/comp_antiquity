import os
import re
import time
import unicodedata
import pandas as pd
import wandb
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
from utils.text import remove_digits, remove_xml_refs  # Assuming these are available
# Hardcoded W&B parameters
WANDB_PROJECT = "greek-cleaning"
WANDB_ENTITY = "sozialismus-au"
WANDB_NAME = "text-cleaning-15apr"
USE_WANDB = True  # Set to False to disable W&B logging

# Define unwanted characters/symbols
# UNWANTED_CHARS = r"[0-9a-zA-Z!#€%&/()?\[\]^_{|}~¶\-\"'“”‘’«»„‚‹›]"
# UNWANTED_CHARS = r"[0-9a-zA-Z!#€%&/()?\[\]^_{|}~¶\-\"'“”‘’«»‹›„‚‟‛「」『』〝〞〟ՙ״؍〃༺༻]"
UNWANTED_CHARS = r"[0-9A-Za-z!*#€%&/()?\[\]^_`{|}~¶<>\-\u0022\u0027\u201C\u201D\u2018\u2019\u00AB\u00BB\u2039\u203A\u201E\u201A\u201F\u201B\u300C\u300D\u300E\u300F\u301D\u301F\u0559\u05F4\u061D\u3003༺༻⋮⋯⸏⸎⸍⸌⸋⸊⸉⸈⸇⸆⸅⸄⸃⸂⸁⸀⸂⸃⸆⏑¯˘〉〈=:†→§]"

# Greek punctuation to preserve
GREEK_PUNCTUATION = {",", ".", ";", "·"}

def normalize_greek_punctuation(text: str) -> str:
    """
    Ensures proper spacing for Greek punctuation:
    - No space before punctuation
    - Exactly one space after punctuation (unless end of string)
    """
    for p in GREEK_PUNCTUATION:
        # Remove space before punctuation
        text = re.sub(rf"\s+{re.escape(p)}", p, text)

        # Ensure exactly one space after punctuation, unless it's end of line or followed by another punctuation
        text = re.sub(rf"{re.escape(p)}(?![\s{re.escape(''.join(GREEK_PUNCTUATION))}]|$)", f"{p} ", text)

    return text


def is_latin(text: str) -> bool:
    """
    Checks if a string contains only Latin characters by inspecting each character.
    """
    try:
        for char in text:  # Iterate character-by-character
            if ord(char) > 127:  # Skip characters outside the ASCII range
                return False
        return True
    except TypeError:
        raise ValueError("Input to is_latin must be a string.")

def contains_latin(text: str) -> bool:
    """
    Check if a string contains any Latin characters.
    """
    try:
        for char in text:  # Iterate character-by-character
            if 65 <= ord(char) <= 90 or 97 <= ord(char) <= 122:  # Latin char range
                return True
        return False
    except TypeError:
        raise ValueError("Input to contains_latin must be a string.")

def clean_text(text: str) -> Tuple[str, Dict]:
    """
    Cleans the input text by normalizing to NFC and then removing unwanted symbols, digits,
    and Latin characters while preserving Greek punctuation.
    
    Parameters:
        text (str): The text to clean.
    
    Returns:
        Tuple[str, Dict]: The cleaned text and stats about changes made
    """
    # Track statistics for logging
    stats = {
        "original_length": len(text),
        "digits_removed": 0,
        "xml_refs_removed": 0,
        "latin_chars_removed": 0,
        "unwanted_chars_removed": 0,
        "multiple_spaces_collapsed": 0
    }
    
    # Normalize text to NFC: this will combine any decomposed diacritics
    text = unicodedata.normalize("NFC", text)
    
    # Remove digits and count them
    text_without_digits = remove_digits(text)
    stats["digits_removed"] = len(text) - len(text_without_digits)
    text = text_without_digits
    
    # Remove XML references and count them
    text_without_xml = remove_xml_refs(text)
    stats["xml_refs_removed"] = len(text) - len(text_without_xml)
    text = text_without_xml
    
    # Remove Latin characters if any exist in the text
    if is_latin(text) or contains_latin(text):
        text_before = text
        text = re.sub(r"[a-zA-Z]", "", text)
        stats["latin_chars_removed"] = len(text_before) - len(text)
    
    # Remove unwanted characters (excluding Greek punctuation)
    text_before = text
    text = re.sub(UNWANTED_CHARS, "", text)
    stats["unwanted_chars_removed"] = len(text_before) - len(text)
    
    # Ensure Greek punctuation is preserved properly with correct spacing
    text = normalize_greek_punctuation(text)

    # Collapse multiple occurrences of any punctuation (e.g., "..." to ".")
    text = re.sub(r"\.{2,}", ".", text)  # Replace multiple full stops with a single one
    text = re.sub(r"\.\s*\.+", ".", text)  # Handles sequences like ". .." -> "."
    
    # Fix single "hanging" full stops after punctuation tokens
    text = re.sub(r"(?<=;\s)\.", "", text)  # e.g., "; ." becomes ";"
    text = re.sub(r"\s+\.(?=\s|$)", ".", text)  # Orphaned full stops
    
    # Collapse whitespace to single spaces
    text_before = text
    text = re.sub(r"\s+", " ", text)
    stats["multiple_spaces_collapsed"] = len(text_before) - len(text)
    
    # Final stats
    stats["final_length"] = len(text.strip())
    stats["reduction_percentage"] = round((1 - stats["final_length"] / stats["original_length"]) * 100, 2) if stats["original_length"] > 0 else 0
    
    return text.strip(), stats

def process_files(src_dir: str, dest_dir: str):
    """
    Processes files listed in the existing index.csv file in src_dir by cleaning them and saving them
    to dest_dir while maintaining the original metadata structure.

    Parameters:
    - src_dir (str): The source directory containing the parsed files and the existing index.csv.
    - dest_dir (str): The destination directory where cleaned files and a new index.csv will be saved.
    """
    # Initialize W&B logging
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_NAME,
        config={
            "src_dir": src_dir,
            "dest_dir": dest_dir,
            "unwanted_chars_pattern": UNWANTED_CHARS,
            "greek_punctuation": list(GREEK_PUNCTUATION)
        }
    )
    
    # Path to the existing index.csv file
    index_csv_path = Path(src_dir) / "index.csv"

    # Ensure the index.csv file exists in the source directory
    if not index_csv_path.exists():
        error_msg = f"index.csv file not found in {src_dir}"
        wandb.log({"error": error_msg})
        raise FileNotFoundError(error_msg)

    # Read the existing index.csv
    index_df = pd.read_csv(index_csv_path)
    
    # Log initial stats
    wandb.log({
        "total_files": len(index_df),
        "start_time": time.time()
    })

    # Keep track of metadata for subdirectory-specific indices
    subdir_indices = {}
    
    # Overall stats for logging
    total_stats = {
        "processed_files": 0,
        "errored_files": 0,
        "total_chars_before": 0,
        "total_chars_after": 0,
        "avg_reduction_percentage": 0,
        "processing_time": 0
    }
    
    start_time = time.time()
    
    # Process each row in the index.csv with progress bar
    for _, row in tqdm(index_df.iterrows(), total=len(index_df), desc="Cleaning files"):
        try:
            # Extract metadata
            src_path = Path(row["dest_path"])  # The existing `dest_path` is the source for cleaning
            source_name = row["source_name"]
            source_id = row["source_id"]
            document_id = row["document_id"]
            title = row.get("title", "")
            author = row.get("author", "")

            # Construct the new destination path in the cleaned folder
            rel_path = src_path.relative_to(src_dir)  # Preserve original structure
            dest_file_path = Path(dest_dir) / rel_path

            # Create destination subfolders as necessary
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Read the file content
            with open(src_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            file_start_time = time.time()

            # Clean the text content and get stats
            cleaned_content, stats = clean_text(content)
            
            file_processing_time = time.time() - file_start_time

            # Save the cleaned content to the designated destination path
            with open(dest_file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_content)

            # Add metadata entry for the subdirectory index
            subdir = dest_file_path.parent
            if subdir not in subdir_indices:
                subdir_indices[subdir] = []
            subdir_indices[subdir].append({
                "src_path": str(src_path),
                "dest_path": str(dest_file_path),
                "source_name": source_name,
                "source_id": source_id,
                "document_id": document_id,
                "title": title,
                "author": author
            })
            
            # Update overall stats
            total_stats["processed_files"] += 1
            total_stats["total_chars_before"] += stats["original_length"]
            total_stats["total_chars_after"] += stats["final_length"]
            total_stats["processing_time"] += file_processing_time
            
            # Log file-specific stats to W&B
            wandb.log({
                "file_stats": {
                    "document_id": document_id,
                    "source_name": source_name,
                    "original_length": stats["original_length"],
                    "final_length": stats["final_length"],
                    "reduction_percentage": stats["reduction_percentage"],
                    "processing_time": file_processing_time,
                    **stats  # Include all the detailed cleaning stats
                }
            })

        except Exception as e:
            error_msg = f"Error processing file {src_path}: {e}"
            print(error_msg)
            total_stats["errored_files"] += 1
            wandb.log({
                "error_file": {
                    "document_id": document_id if 'document_id' in locals() else "unknown",
                    "source_path": str(src_path) if 'src_path' in locals() else "unknown",
                    "error_message": str(e)
                }
            })

    # Calculate final statistics
    total_stats["avg_reduction_percentage"] = round(
        (1 - total_stats["total_chars_after"] / total_stats["total_chars_before"]) * 100, 2
    ) if total_stats["total_chars_before"] > 0 else 0
    total_stats["total_runtime"] = time.time() - start_time
    
    # Save the subdirectory-specific indices
    for subdir, metadata in subdir_indices.items():
        subdir_index_csv_path = subdir / "index.csv"
        subdir_index_df = pd.DataFrame(metadata)
        subdir_index_df.to_csv(subdir_index_csv_path, index=False, encoding="utf-8")
        print(f"Subdirectory index saved to {subdir_index_csv_path}")

    # Save the global new metadata to the index.csv in the destination directory
    cleaned_index_csv_path = Path(dest_dir) / "index.csv"
    cleaned_index_data = [row for metadata in subdir_indices.values() for row in metadata]
    cleaned_index_df = pd.DataFrame(cleaned_index_data)
    cleaned_index_df.to_csv(cleaned_index_csv_path, index=False, encoding="utf-8")
    print(f"Global index metadata saved to {cleaned_index_csv_path}")
    
    # Log final summary stats to W&B
    wandb.log({
        "summary": {
            "total_files_processed": total_stats["processed_files"],
            "error_files": total_stats["errored_files"],
            "success_rate": round((total_stats["processed_files"] / len(index_df)) * 100, 2),
            "total_characters_before": total_stats["total_chars_before"],
            "total_characters_after": total_stats["total_chars_after"],
            "overall_reduction_percentage": total_stats["avg_reduction_percentage"],
            "total_runtime_seconds": total_stats["total_runtime"],
            "avg_time_per_file": total_stats["processing_time"] / total_stats["processed_files"] if total_stats["processed_files"] > 0 else 0
        }
    })
    
    # Create and log a summary artifact to W&B
    summary_table = wandb.Table(columns=["Metric", "Value"])
    summary_table.add_data("Total Files", len(index_df))
    summary_table.add_data("Successfully Processed", total_stats["processed_files"])
    summary_table.add_data("Files with Errors", total_stats["errored_files"])
    summary_table.add_data("Success Rate (%)", round((total_stats["processed_files"] / len(index_df)) * 100, 2))
    summary_table.add_data("Total Input Characters", total_stats["total_chars_before"])
    summary_table.add_data("Total Output Characters", total_stats["total_chars_after"])
    summary_table.add_data("Total Reduction (%)", total_stats["avg_reduction_percentage"])
    summary_table.add_data("Total Runtime (seconds)", total_stats["total_runtime"])
    summary_table.add_data("Average Processing Time Per File (seconds)", 
                           total_stats["processing_time"] / total_stats["processed_files"] if total_stats["processed_files"] > 0 else 0)
    
    wandb.log({"cleaning_summary": summary_table})
    
    # Finish the W&B run
    run.finish()


if __name__ == "__main__":
    # Define source and destination directories
    SRC_DIR = "dat/greek/parsed_data"
    DEST_DIR = "dat/greek/cleaned_parsed_data"

    print("Starting text cleaning process...")
    process_files(SRC_DIR, DEST_DIR)
    print("Text cleaning process complete.")
