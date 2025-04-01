import re
import csv
import json
import logging
import sys
from pathlib import Path

# Set base export directory
BASE_EXPORT_DIR = Path("dat/export/bjarke_test")
LOG_DIR = BASE_EXPORT_DIR / "logs"
LOG_FILE = LOG_DIR / "conllu_extract_process.log"

def setup_logging():
    """Sets up logging to a specific file while also printing logs to console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    logging.info("Logging initialized.")

def extract_conllu_annotation(conllu_file, annotation_type, output_dir):
    """
    Extracts token ID, form, and a specific annotation (lemma, upos, or ner)
    from a CoNLL-U file and writes the output to a CSV file in the respective
    annotation directory.
    """
    extracted_data = []
    with open(conllu_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            columns = line.split("\t")
            if len(columns) < 10:
                continue

            token_id = columns[0]  # Using global_id as the token id
            form = columns[1]
            
            if annotation_type == "lemma":
                annotation = columns[2]  # Lemma column
            elif annotation_type == "upos":
                annotation = columns[3]  # UPOS column
            elif annotation_type == "ner":
                misc = columns[9]
                ner_match = re.search(r"NER=([^\s|]+)", misc)
                annotation = ner_match.group(1) if ner_match else "O"
            else:
                logging.warning(f"Unknown annotation type: {annotation_type}")
                return
            
            extracted_data.append({
                "ID": token_id,
                "FORM": form,
                annotation_type.upper(): annotation
            })
    
    output_file = output_dir / f"{conllu_file.stem}-{annotation_type}.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["ID", "FORM", annotation_type.upper()]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extracted_data)
    
    logging.info(f"Extracted {annotation_type} annotations to {output_file}")

def process_all_conllu_files(base_dir):
    """Recursively finds all .conllu files and extracts annotations for each."""
    for source_dir in base_dir.iterdir():
        if not source_dir.is_dir():
            continue
        
        for id_dir in source_dir.iterdir():
            if not id_dir.is_dir():
                continue
            
            annotations_folder = id_dir / "annotations"
            conllu_file = annotations_folder / f"{id_dir.name}.conllu"
            
            if not conllu_file.exists():
                logging.warning(f"Missing .conllu file in {annotations_folder}")
                continue
            
            for annotation_type in ["lemma", "upos", "ner"]:
                extract_conllu_annotation(conllu_file, annotation_type, annotations_folder)

if __name__ == "__main__":
    setup_logging()
    process_all_conllu_files(BASE_EXPORT_DIR)
