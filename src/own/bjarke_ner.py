import os
import json
import spacy
import logging
import gc
import sys
from pathlib import Path

# Set base export directory
BASE_EXPORT_DIR = Path("dat/export/bjarke_test")
LOG_DIR = BASE_EXPORT_DIR / "logs"
LOG_FILE = LOG_DIR / "ner_process.log"

CHUNK_SIZE = 500000  # Process text in chunks of 500 KB to avoid memory issues

def setup_logging():
    """Sets up logging to a specific file while also printing logs to console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler to write logs to file
    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    # Create console handler to display logs in terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    logging.info("Logging initialized.")

# Load the NER model and ensure it loads correctly
try:
    nlp = spacy.load("grc_ner_trf")
    logging.info("Loaded grc_ner_trf model.")
except Exception as e:
    logging.error(f"Error loading grc_ner_trf model: {e}")
    exit(1)

def process_ner_file(file_path, output_path, doc_name):
    """Processes a single text file, applying NER and saving results in NDJSON format."""
    if output_path.exists():
        logging.info(f"Skipping already processed file: {output_path}")
        return

    global_id = 1
    try:
        logging.info(f"Processing NER for file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f, open(output_path, "a", encoding="utf-8") as out_f:
            while True:
                text = f.read(CHUNK_SIZE)
                if not text:
                    break
                
                doc = nlp(text)
                for token in doc:
                    if token.text.strip():
                        token_data = {
                            "global_id": global_id,
                            "id": token.i + 1,  # Ensures continuous unique ID across chunks
                            "text": token.text,
                            "ner": token.ent_type_ if token.ent_type_ else "O",
                            "doc_name": doc_name
                        }
                        out_f.write(json.dumps(token_data, ensure_ascii=False) + "\n")
                        global_id += 1
        
        logging.info(f"Successfully processed NER: {file_path}")
    except Exception as e:
        logging.error(f"Error processing NER for {file_path}: {e}")
    finally:
        gc.collect()

def process_ner_dataset():
    """
    Iterates over the BASE_EXPORT_DIR, finding text files in 'texts/' folders,
    processing them with the NER model, and saving the output in 'annotations/'.
    """
    for source_dir in BASE_EXPORT_DIR.iterdir():
        if not source_dir.is_dir():
            continue
        for id_dir in source_dir.iterdir():
            if not id_dir.is_dir():
                continue
            texts_folder = id_dir / "texts"
            annotations_folder = id_dir / "annotations"
            if not texts_folder.exists():
                logging.warning(f"Missing texts folder: {texts_folder}")
                continue
            annotations_folder.mkdir(exist_ok=True)
            for text_file in texts_folder.glob("*-joined.txt"):
                output_file = annotations_folder / (text_file.stem.replace("-joined", "") + "-ner.json")
                process_ner_file(text_file, output_file, text_file.stem)

if __name__ == "__main__":
    setup_logging()
    process_ner_dataset()
    print(f"NER processing completed. Check log file: {LOG_FILE}")
