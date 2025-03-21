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

def process_files(src: str, dest: str, index_file: str):
    """
    Processes all .txt files in the source directory, cleans their content, 
    and saves them in the destination directory while preserving folder structure.

    Parameters:
    src (str): The source directory containing .txt files.
    dest (str): The destination directory to save cleaned files.
    index_file (str): Path to the index.csv file from the source directory.
    """

    # Ensure the destination directory exists
    Path(dest).mkdir(parents=True, exist_ok=True)
    
    # Prepare a new index list to mimic the structure of index.csv
    new_index_data = []

    # Iterate through all subdirectories and files
    for root, _, files in os.walk(src):
        for file in files:
            if file.endswith(".txt"):
                # Full source path
                src_file_path = os.path.join(root, file)

                # Mimic subdirectory structure in destination
                sub_dir = root.replace(src, "").lstrip(os.sep)
                dest_dir_path = os.path.join(dest, sub_dir)
                Path(dest_dir_path).mkdir(parents=True, exist_ok=True)

                # Destination file path
                dest_file_path = os.path.join(dest_dir_path, file)

                # Read and clean the file content
                with open(src_file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    cleaned_content = clean_text(content)

                # Write the cleaned content to the destination file
                with open(dest_file_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_content)

                # Update the new index
                new_index_data.append({
                    "document_id": Path(src_file_path).stem,  # Use filename without extension
                    "src_path": src_file_path,
                    "dest_path": dest_file_path
                })

    # Save the new index.csv file
    new_index_path = os.path.join(dest, "index.csv")
    new_index_df = pd.DataFrame(new_index_data)
    new_index_df.to_csv(new_index_path, index=False)
    print(f"Saved index.csv to {new_index_path}")

if __name__ == "__main__":
    # Define source and destination directories
    SRC_DIR = "dat/greek/parsed_data"
    DEST_DIR = "dat/greek/cleaned_parsed_data"
    
    # Path to the existing index.csv file
    INDEX_FILE = os.path.join(SRC_DIR, "index.csv")

    print("Starting text cleaning process...")
    process_files(SRC_DIR, DEST_DIR, INDEX_FILE)
    print("Text cleaning process complete.")
