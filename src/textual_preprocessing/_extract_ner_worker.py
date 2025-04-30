# _extract_ner_worker.py
import argparse
import csv
import json
import logging
import os
import sys
import traceback

import argparse
import csv
import os
import re
import shutil
import subprocess
import time
import tempfile
import traceback
import logging
import signal
import sys
import json
import glob
from collections import Counter
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import tqdm
import wandb
import spacy
from spacy.tokens import DocBin

# Configure basic logging within the script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] NER-Worker: %(message)s',
    stream=sys.stderr
)

def extract_ner_data(
    docbin_path: str,
    output_csv_ner: str,
    output_ner_tags: str, # Path for the simple tag list file
    output_stats_json: str,
    ner_model_name: str,
    old_id_str: str,
    new_id_str: str,
):
    """Loads NER model, processes docbin, writes NER CSV, tags list, and stats."""
    logging.info(f"Starting NER extraction for Doc ID: {old_id_str} -> {new_id_str}")
    logging.info(f"NER DocBin Path: {docbin_path}")
    logging.info(f"NER Model: {ner_model_name}")
    logging.info(f"Output CSV: {output_csv_ner}")
    logging.info(f"Output Tags File: {output_ner_tags}")
    logging.info(f"Output Stats JSON: {output_stats_json}")

    # Ensure output directories exist
    dirs_to_check=set([os.path.dirname(p) for p in [output_csv_ner, output_ner_tags, output_stats_json] if p])
    for d in dirs_to_check:
        if d:
            try:
                 os.makedirs(d, exist_ok=True); logging.info(f"Ensured directory exists: {d}")
            except OSError as e: logging.error(f"Failed create dir {d}: {e}"); sys.exit(1)

    try:
        # --- Load Model and DocBin ---
        logging.info(f"Loading NER model '{ner_model_name}'...")
        nlp = spacy.load(ner_model_name)
        logging.info(f"Loading DocBin from {docbin_path}...")
        doc_bin = DocBin().from_disk(docbin_path)
        docs = list(doc_bin.get_docs(nlp.vocab))
        if not docs:
            logging.error(f"NER DocBin is empty or failed load: {docbin_path}")
            sys.exit(1)
        doc = docs[0]; num_tokens = len(doc)
        logging.info(f"Loaded doc with {num_tokens} tokens. Found {len(doc.ents)} entities.")

        # --- Extract NER CSV and Tags List ---
        ner_stats={'total_tokens': num_tokens, 'ner_tokens': 0, 'o_tokens': 0}
        tags_list = []
        logging.info(f"Writing NER CSV to {output_csv_ner} and tags to {output_ner_tags}...")
        try:
            with open(output_csv_ner,'w',encoding='utf-8',newline='') as fner, \
                 open(output_ner_tags,'w',encoding='utf-8') as ftags:
                # Use QUOTE_ALL for safety
                wn = csv.writer(fner, quoting=csv.QUOTE_ALL)
                wn.writerow(['ID','TOKEN','NER'])

                for i, t in enumerate(doc):
                    tid = i + 1
                    ttxt = str(t.text)
                    nt = t.ent_type_ if t.ent_type_ else 'O'

                    if nt != 'O':
                        ner_stats['ner_tokens'] += 1
                    else:
                        ner_stats['o_tokens'] += 1

                    wn.writerow([tid, ttxt, nt])
                    tags_list.append(nt) # Append the tag (e.g., 'O', 'PERSON') to the list

                # Write all tags to the tags file, one per line
                ftags.write('\n'.join(tags_list))
            logging.info(f"Finished writing NER CSV and tags file. NER tokens: {ner_stats['ner_tokens']}, O tokens: {ner_stats['o_tokens']}.")

        except Exception as csv_e:
            logging.error(f"Failed during NER CSV/Tags writing: {csv_e}", exc_info=True)
            sys.exit(1)

        # --- Write NER Stats Summary ---
        logging.info(f"Writing NER stats summary to {output_stats_json}...")
        try:
            ner_percentage = (ner_stats['ner_tokens'] / num_tokens * 100) if num_tokens > 0 else 0
            summary_data = {
                "doc_id_original": old_id_str, "doc_id_new": new_id_str,
                "docbin_path": docbin_path, # Original path for reference
                "ner_model_used": ner_model_name,
                "total_tokens": ner_stats['total_tokens'],
                "tokens_with_ner": ner_stats['ner_tokens'],
                "tokens_without_ner": ner_stats['o_tokens'],
                "ner_percentage": round(ner_percentage, 2)
            }
            with open(output_stats_json,'w',encoding='utf-8') as fsum:
                json.dump(summary_data, fsum, indent=2)
            logging.info("Finished writing NER stats summary.")

        except Exception as e:
            # Log failure but don't necessarily exit - summary is non-critical
            logging.warning(f"Failed write NER summary file: {e}", exc_info=True)

    except Exception as e:
        logging.error(f"Unhandled exception in NER worker script: {e}", exc_info=True)
        sys.exit(1)

    logging.info(f"NER extraction script finished successfully for Doc ID: {old_id_str} -> {new_id_str}.")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER Worker Script")
    parser.add_argument("--docbin-path", required=True, help="Path to input DocBin file.")
    parser.add_argument("--output-csv-ner", required=True, help="Path for output NER CSV file.")
    parser.add_argument("--output-ner-tags", required=True, help="Path for output NER tags text file.")
    parser.add_argument("--output-stats-json", required=True, help="Path for output NER stats JSON file.")
    parser.add_argument("--ner-model-name", required=True, help="Name of the spaCy NER model to load.")
    parser.add_argument("--old-id", required=True, help="Original document ID.")
    parser.add_argument("--new-id", required=True, help="New document ID (Sort ID).")

    args = parser.parse_args()

    extract_ner_data(
        docbin_path=args.docbin_path,
        output_csv_ner=args.output_csv_ner,
        output_ner_tags=args.output_ner_tags,
        output_stats_json=args.output_stats_json,
        ner_model_name=args.ner_model_name,
        old_id_str=args.old_id,
        new_id_str=args.new_id,
    )
