from pathlib import Path
import unicodedata
import logging
import csv

# Configure logging
output_directory = Path("output")  # Change this to your desired output folder
output_directory.mkdir(exist_ok=True)  # Create the directory if it doesn't exist

log_file_path = output_directory / "normalization_issues.log"
csv_file_path = output_directory / "normalization_report.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

def detect_unicode_normalization(text):
    """
    Detects whether text is normalized in NFC or NFD form.
    Returns 'NFC', 'NFD', or 'MIXED' if inconsistent.
    """
    if not isinstance(text, str) or not text.strip():
        return "EMPTY"

    normalized_nfc = unicodedata.normalize("NFC", text)
    normalized_nfd = unicodedata.normalize("NFD", text)

    if text == normalized_nfc:
        return "NFC"
    elif text == normalized_nfd:
        return "NFD"
    else:
        return "MIXED"

def analyze_text_files(directory):
    """
    Recursively scans a directory for .txt files, checks Unicode normalization,
    and logs results in a CSV report and log file.
    """
    directory = Path(directory)
    results = []
    total_nfc = total_nfd = total_mixed = 0

    for file_path in directory.rglob("*.txt"):  # Recursively find all .txt files
        try:
            with file_path.open("r", encoding="utf-8") as file:
                text = file.read()
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            continue

        norm_type = detect_unicode_normalization(text)

        if norm_type == "NFC":
            total_nfc += 1
        elif norm_type == "NFD":
            total_nfd += 1
        elif norm_type == "MIXED":
            total_mixed += 1
            logging.warning(f"Inconsistent normalization in {file_path}")

        results.append([str(file_path), norm_type, text[:100]])  # Store first 100 chars for context

    # Print summary
    logging.info(f"Total files checked: {len(results)}")
    logging.info(f"NFC: {total_nfc} | NFD: {total_nfd} | Mixed: {total_mixed}")

    # Write results to CSV
    with csv_file_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Normalization Type", "Sample Text"])
        writer.writerows(results)

    print(f"✅ Analysis complete! See '{csv_file_path}' and '{log_file_path}' for details.")

# Run the analysis
directory_path = "dat/greek/cleaned_parsed_data/"  # Replace with the actual folder containing .txt files
analyze_text_files(directory_path)
