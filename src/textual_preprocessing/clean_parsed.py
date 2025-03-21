import os
import re
import pandas as pd
from pathlib import Path
from utils.text import remove_digits, remove_xml_refs  # Assuming these are available

# Define unwanted characters/symbols
UNWANTED_CHARS = r"[0-9a-zA-Z!#€%&/()?\[\]^_`{|}~¶]"

# Greek punctuation
GREEK_PUNCTUATION = {",", ".", ";", "·", "„", "“", "…"}


def is_latin(text: str) -> bool:
    """
    Checks if a string contains only Latin characters by inspecting each character.
    Parameters:
        text (str): The input string.
    Returns:
        bool: True if the text contains only Latin characters, False otherwise.
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
    Parameters:
        text (str): The input string.
    Returns:
        bool: True if the string contains any Latin characters, False otherwise.
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
    Cleans the input text by removing unwanted symbols, digits, 
    and Latin characters while preserving Greek punctuation.
    
    Parameters:
    text (str): The text to clean.

    Returns:
    str: The cleaned text.
    """
    # Remove digits
    text = remove_digits(text)
    
    # Remove XML references
    text = remove_xml_refs(text)
    
    # Remove Latin characters if the text contains them
    # if is_latin(text) or contains_latin(text):
    #     text = re.sub(r"[a-zA-Z]", "", text)

    if is_latin(text) or contains_latin(text):
        text = re.sub(r"[a-zA-Z]", "", text)
    
    # Remove unwanted characters (excluding Greek punctuation)
    text = re.sub(UNWANTED_CHARS, "", text)
    

    # Collapse multiple occurrences of any punctuation (e.g., "..." to ".")
    text = re.sub(r"\.{2,}", ".", text)  # Replace multiple full stops with a single one
    text = re.sub(r"\.\s*\.+", ".", text)  # Handles sequences like ". .." -> "."

    # Fix single "hanging" full stops after punctuation or semantically meaningful tokens
    text = re.sub(r"(?<=;\s)\.", "", text)  # Removes a single hanging "." after a semicolon (e.g., "; .")
    text = re.sub(r"\s+\.(?=\s|$)", ".", text)  # Fixes cases where a full stop is orphaned (e.g., " ; . ")


    # Ensure Greek punctuation is preserved properly with correct spacing
    for punctuation in GREEK_PUNCTUATION:
        # Use regex to ensure spacing around punctuation
        text = re.sub(rf"(?<!\s){re.escape(punctuation)}(?!\s)", f" {punctuation} ", text)

    # Collapse whitespace to single spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def process_files(src_dir: str, dest_dir: str, index_csv: str):
    """
    Processes files in src_dir, cleans them, and saves to dest_dir with updated identifiers.

    Parameters:
    - src_dir (str): Source directory containing parsed files organized by subfolders.
    - dest_dir (str): Destination directory for cleaned files.
    - index_csv (str): Path to the CSV file where index information is saved.
    """

    # Prepare the index data
    index_data = []

    # Walk through the source directory
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".txt"):  # Process only .txt files
                # Define source and destination paths
                src_file_path = Path(root) / file
                rel_path = src_file_path.relative_to(src_dir)
                dest_subfolder = Path(dest_dir) / rel_path.parent
                dest_file_path = dest_subfolder / file

                # Extract corpus name from the relative path (e.g., "parsed_data/perseus")
                corpus_name = rel_path.parts[0]  # This assumes the first subfolder is the corpus
                document_id = f"{corpus_name}.{file.replace('.txt', '')}"  # Updated document_id
                source_id = file.replace(".txt", "")  # Source ID (file name without extension)

                # Ensure the destination directory exists
                dest_subfolder.mkdir(parents=True, exist_ok=True)

                # Read, clean, and save the file
                try:
                    with open(src_file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    cleaned_content = clean_text(content)  # Call your cleaning function here

                    with open(dest_file_path, "w", encoding="utf-8") as f:
                        f.write(cleaned_content)

                    # Append metadata to the index
                    index_data.append({
                        "document_id": document_id,
                        "source_id": source_id,
                        "src_path": str(src_file_path),
                        "dest_path": str(dest_file_path)
                    })

                except Exception as e:
                    print(f"Error processing file: {src_file_path}, Error: {e}")

    # Save index.csv
    index_df = pd.DataFrame(index_data)
    index_df.to_csv(index_csv, index=False)
    print(f"Index saved to {index_csv}")

if __name__ == "__main__":
    # Define source and destination directories
    SRC_DIR = "dat/greek/parsed_data"
    DEST_DIR = "dat/greek/cleaned_parsed_data"
    
    # Path to the existing index.csv file
    INDEX_FILE = os.path.join(SRC_DIR, "index.csv")

    print("Starting text cleaning process...")
    process_files(SRC_DIR, DEST_DIR, INDEX_FILE)
    print("Text cleaning process complete.")
