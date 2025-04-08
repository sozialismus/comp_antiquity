import os
import re
import unicodedata
import pandas as pd
from pathlib import Path
from utils.text import remove_digits, remove_xml_refs  # Assuming these are available

# Define unwanted characters/symbols
# UNWANTED_CHARS = r"[0-9a-zA-Z!#€%&/()?\[\]^_{|}~¶\-\"'“”‘’«»„‚‹›]"
# UNWANTED_CHARS = r"[0-9a-zA-Z!#€%&/()?\[\]^_{|}~¶\-\"'“”‘’«»‹›„‚‟‛「」『』〝〞〟ՙ״؍〃༺༻]"
UNWANTED_CHARS = r"[0-9A-Za-z!#€%&/()?\[\]^_`{|}~¶<>\-\u0022\u0027\u201C\u201D\u2018\u2019\u00AB\u00BB\u2039\u203A\u201E\u201A\u201F\u201B\u300C\u300D\u300E\u300F\u301D\u301F\u0559\u05F4\u061D\u3003༺༻⋮⋯⸏⸎⸍⸌⸋⸊⸉⸈⸇⸆⸅⸄⸃⸂⸁⸀⏑¯˘〉〈=:]"

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

def clean_text(text: str) -> str:
    """
    Cleans the input text by normalizing to NFC and then removing unwanted symbols, digits,
    and Latin characters while preserving Greek punctuation.
    
    Parameters:
        text (str): The text to clean.
    
    Returns:
        str: The cleaned text.
    """
    # Normalize text to NFC: this will combine any decomposed diacritics
    text = unicodedata.normalize("NFC", text)
    
    # Remove digits
    text = remove_digits(text)
    
    # Remove XML references
    text = remove_xml_refs(text)
    
    # Remove Latin characters if any exist in the text
    if is_latin(text) or contains_latin(text):
        text = re.sub(r"[a-zA-Z]", "", text)
    
    # Remove unwanted characters (excluding Greek punctuation)
    text = re.sub(UNWANTED_CHARS, "", text)
    
    # Ensure Greek punctuation is preserved properly with correct spacing
    text = normalize_greek_punctuation(text)

    # Collapse multiple occurrences of any punctuation (e.g., "..." to ".")
    text = re.sub(r"\.{2,}", ".", text)  # Replace multiple full stops with a single one
    text = re.sub(r"\.\s*\.+", ".", text)  # Handles sequences like ". .." -> "."
    
    # Fix single "hanging" full stops after punctuation tokens
    text = re.sub(r"(?<=;\s)\.", "", text)  # e.g., "; ." becomes ";"
    text = re.sub(r"\s+\.(?=\s|$)", ".", text)  # Orphaned full stops
    
    
    # Collapse whitespace to single spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def process_files(src_dir: str, dest_dir: str):
    """
    Processes files listed in the existing index.csv file in src_dir by cleaning them and saving them
    to dest_dir while maintaining the original metadata structure.

    Parameters:
    - src_dir (str): The source directory containing the parsed files and the existing index.csv.
    - dest_dir (str): The destination directory where cleaned files and a new index.csv will be saved.
    """
    # Path to the existing index.csv file
    index_csv_path = Path(src_dir) / "index.csv"

    # Ensure the index.csv file exists in the source directory
    if not index_csv_path.exists():
        raise FileNotFoundError(f"index.csv file not found in {src_dir}")

    # Read the existing index.csv
    index_df = pd.read_csv(index_csv_path)

    # Keep track of metadata for subdirectory-specific indices
    subdir_indices = {}

    # Process each row in the index.csv
    for _, row in index_df.iterrows():
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

            # Read, clean, and save the file
            with open(src_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Clean the text content
            cleaned_content = clean_text(content)

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

        except Exception as e:
            print(f"Error processing file {src_path}: {e}")

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


if __name__ == "__main__":
    # Define source and destination directories
    SRC_DIR = "dat/greek/parsed_data"
    DEST_DIR = "dat/greek/cleaned_parsed_data"

    print("Starting text cleaning process...")
    process_files(SRC_DIR, DEST_DIR)
    print("Text cleaning process complete.")
