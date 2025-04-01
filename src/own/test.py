import os
import json
import spacy
import logging
import gc
import re
from pathlib import Path

def setup_logging():
    logging.basicConfig(
        filename="proiel_process_log.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"  # Overwrites the log file on each run
    )
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console_handler)
    
    logging.info("Logging initialized.")

# Load the PROIEL model
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

if __name__ == "__main__":
    setup_logging()

    # Define file paths
    input_file = "dat/export/bjarke_test/perseus/321/texts/321-joined.txt"  # Replace with your actual file path
    output_file = "dat/export/bjarke_test/perseus/321/annotations/321-joined_test.txt"  # Replace with desired output path
    doc_name = Path(input_file).stem  # Extracts the filename without extension

    process_text_chunked(input_file, output_file, doc_name)

    print("Processing completed. Check proiel_process_log.log for details.")
