import os
import json
import spacy
import logging
import gc
import sys
import re
from pathlib import Path

# Set base export directory
BASE_EXPORT_DIR = Path("dat/export/bjarke_test")
LOG_DIR = BASE_EXPORT_DIR / "logs"
LOG_FILE = LOG_DIR / "proiel_process.log"

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

# Load the PROIEL model and ensure the parser is enabled
try:
    nlp = spacy.load("grc_proiel_trf")
    nlp.enable_pipe("parser")
    logging.info("Loaded grc_proiel_trf model.")
except Exception as e:
    logging.error(f"Error loading grc_proiel_trf model: {e}")
    exit(1)

def process_sentence(sent, out_f, global_id, doc_name):
    """Processes a single sentence, writes token annotations in JSON Lines, and updates global_id."""
    for token in sent:
        if token.text.strip():
            head = token.head.i + 1 if token.head != token else 0
            dep = token.dep_
            deps = f"{head}:{dep}" if head > 0 else "_"
            token_data = {
                "global_id": global_id,
                "id": token.i + 1,
                "text": token.text,
                "lemma": token.lemma_,
                "upos": token.pos_,
                "xpos": token.tag_,
                "feats": str(token.morph) if token.morph else "_",
                "head": head,
                "dep": dep,
                "deps": deps,
                "misc": "_",
                "doc_name": doc_name
            }
            out_f.write(json.dumps(token_data, ensure_ascii=False) + "\n")
            global_id += 1
    return global_id

def process_text_chunked(file_path, output_path, doc_name, chunk_size=1024*1024):  # 1MB chunks
    """Processes a large file by reading it in chunks, splitting into sentences, and processing them."""
    global_id = 1
    try:
        logging.info(f"Processing file in chunks: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            text_chunks = []
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                text_chunks.append(chunk)
        
        # Join chunks and use regex to split into sentences (ensuring not to break sentences mid-chunk)
        text = " ".join(text_chunks).strip()
        sentences = re.split(r'(?<=[.!?])\s+', text)

        with open(output_path, "w", encoding="utf-8") as out_f:
            for sent_doc in nlp.pipe(sentences, batch_size=10):  # ✅ Process in streaming mode
                for sent in sent_doc.sents:
                    global_id = process_sentence(sent, out_f, global_id, doc_name)

        logging.info(f"Successfully processed: {file_path}")

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")

    finally:
        gc.collect()  # ✅ Clean up memory after each file

def process_proiel_dataset():
    """
    Iterates over the BASE_EXPORT_DIR; for each source and id-specific subfolder,
    finds text files in the 'texts/' folder (i.e., files ending with '-joined.txt'),
    processes them with the PROIEL model (reading the full file as a single string),
    and saves the output in the corresponding 'annotations/' folder as <id>-proiel.json.
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
                # Construct the output file name by replacing '-joined.txt' with '-proiel.json'
                output_file = annotations_folder / (text_file.stem.replace("-joined", "") + "-proiel.json")
                process_text_chunked(text_file, output_file, text_file.stem)

if __name__ == "__main__":
    setup_logging()
    process_proiel_dataset()
    print(f"PROIEL processing completed. Check log file: {LOG_FILE}")
