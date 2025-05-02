# generate_reorganization_index.py
import argparse
import logging
import sys
import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import tqdm

# --- Configure standard logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
script_logger = logging.getLogger("ReorgIndexGenerator")

# --- Helper function (copied/adapted from main reorganization script) ---
# This function determines the *expected* paths based on the conventions
# used in the main reorganization script.
def get_expected_output_paths(output_base_dir: str, corpus_prefix: str, new_id: str) -> Dict[str, str]:
    """Calculates the expected standard output paths for a reorganized document."""
    abs_output_base_dir = os.path.abspath(output_base_dir)
    # Structure: output_base_dir / corpus_prefix / new_id / {texts|annotations} / files
    output_dir = os.path.join(abs_output_base_dir, corpus_prefix, new_id)
    texts_dir = os.path.join(output_dir, "texts")
    annotations_dir = os.path.join(output_dir, "annotations")
    return {
        "output_dir": output_dir,
        "texts_dir": texts_dir,
        "annotations_dir": annotations_dir,
        "output_txt_joined": os.path.join(texts_dir, f"{new_id}-joined.txt"),
        "output_txt_fullstop": os.path.join(texts_dir, f"{new_id}-fullstop.txt"),
        "output_csv_lemma": os.path.join(annotations_dir, f"{new_id}-lemma.csv"),
        "output_csv_upos": os.path.join(annotations_dir, f"{new_id}-upos.csv"),
        "output_csv_stop": os.path.join(annotations_dir, f"{new_id}-stop.csv"),
        "output_csv_dot": os.path.join(annotations_dir, f"{new_id}-dot.csv"),
        "output_csv_ner": os.path.join(annotations_dir, f"{new_id}-ner.csv"),
        "output_conllu": os.path.join(annotations_dir, f"{new_id}-conllu.conllu"),
        "output_ner_tags": os.path.join(annotations_dir, f"{new_id}-ner_tags.txt"),
        "source_original_txt": os.path.join(texts_dir, f"{new_id}-original.txt"), # Copied source
        "ner_mismatch_info": os.path.join(annotations_dir, f"{new_id}_ner_mismatch_info.json"),
        "ner_stats_file": os.path.join(annotations_dir, f"{new_id}_ner_stats.json"),
        "ner_error_info": os.path.join(annotations_dir, f"{new_id}_ner_error_info.json"),
    }

# --- Helper function (copied/adapted from main reorganization script) ---
def parse_csv_mapping(csv_path: str) -> List[Dict[str, str]]:
    """
    Parses the mapping CSV into a list of dictionaries.
    Returns list of {'old_id': str, 'new_id': str}.
    Logs warnings for duplicates or invalid entries but continues.
    """
    mappings_list = []
    processed_old_ids = set()
    abs_csv_path = os.path.abspath(csv_path)
    script_logger.info(f"Loading mapping file: {abs_csv_path}")
    try:
        # Use low_memory=False for potentially large files with mixed types
        df = pd.read_csv(abs_csv_path, dtype={'document_id': str, 'sort_id': str}, low_memory=False)
        id_col = 'document_id'
        sort_col = 'sort_id'

        if id_col not in df.columns:
            script_logger.error(f"Mapping file '{abs_csv_path}' missing required column '{id_col}'.")
            return []
        if sort_col not in df.columns:
            script_logger.error(f"Mapping file '{abs_csv_path}' missing required column '{sort_col}'.")
            return []

        duplicates = 0
        invalid_sort_ids = 0
        missing_sort_ids = 0
        total_rows = len(df)

        for index, row in tqdm.tqdm(df.iterrows(), total=total_rows, desc="Parsing mapping CSV", unit="row"):
            old_id = row[id_col]
            new_id_val = row[sort_col]

            # Basic validation
            if pd.isna(old_id) or not str(old_id).strip():
                script_logger.debug(f"Skipping row {index+2}: Missing or empty document_id.")
                continue
            old_id = str(old_id).strip()

            if pd.isna(new_id_val) or not str(new_id_val).strip():
                script_logger.warning(f"Missing sort_id (NaN/None/Empty) for doc '{old_id}' in '{abs_csv_path}' (row {index+2}). Skipping.")
                missing_sort_ids += 1
                continue

            new_id = str(new_id_val).strip()
            # Optional: Add more specific validation for new_id format if needed (e.g., must be numeric)
            # try:
            #     int(float(new_id)) # Check if it can be interpreted as an integer
            # except (ValueError, TypeError):
            #      script_logger.warning(f"Invalid non-numeric sort_id '{new_id}' for doc '{old_id}' in '{abs_csv_path}' (row {index+2}). Skipping.")
            #      invalid_sort_ids += 1
            #      continue

            if old_id in processed_old_ids:
                script_logger.warning(f"Duplicate document_id '{old_id}' found in mapping '{abs_csv_path}' (row {index+2}). Skipping subsequent occurrence.")
                duplicates += 1
                continue

            mappings_list.append({'old_id': old_id, 'new_id': new_id})
            processed_old_ids.add(old_id)

        if duplicates > 0: script_logger.warning(f"Found and skipped {duplicates} duplicate document IDs in {abs_csv_path}.")
        #if invalid_sort_ids > 0: script_logger.warning(f"Found and skipped {invalid_sort_ids} invalid sort IDs in {abs_csv_path}.")
        if missing_sort_ids > 0: script_logger.warning(f"Found and skipped {missing_sort_ids} missing sort IDs in {abs_csv_path}.")
        script_logger.info(f"Successfully loaded {len(mappings_list)} unique mappings from: {abs_csv_path}")

    except FileNotFoundError:
        script_logger.error(f"Mapping file not found at '{abs_csv_path}'")
        return []
    except pd.errors.EmptyDataError:
        script_logger.error(f"Mapping file is empty: '{abs_csv_path}'")
        return []
    except Exception as e:
        script_logger.error(f"Error parsing mapping file '{abs_csv_path}': {type(e).__name__}: {e}", exc_info=True)
        return []
    return mappings_list

def create_parser() -> argparse.ArgumentParser:
    """Creates parser for the reorganization index generation utility."""
    parser = argparse.ArgumentParser(
        description="Generates a master index of reorganized corpus files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mapping-csv", type=Path, required=True,
        help="Path to the CSV mapping 'document_id' (old_id) to 'sort_id' (new_id) used for reorganization."
    )
    parser.add_argument(
        "--output-base-dir", type=Path, required=True,
        help="Path to the base directory WHERE the reorganized files were saved by the main script."
    )
    parser.add_argument(
        "--output-index-csv", type=Path, required=True,
        help="Path for the output master index CSV file."
    )
    parser.add_argument(
        "--key-file-check", type=str, default="output_conllu",
        choices=list(get_expected_output_paths("base", "prefix", "id").keys()), # Get valid keys
        help="Which expected output file's existence determines 'complete' status. "
             "'output_conllu' is a good default as it's generated late in the main worker."
    )
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    script_logger.info("--- Generating Reorganization Master Index ---")
    script_logger.info(f"Mapping CSV: {args.mapping_csv.resolve()}")
    script_logger.info(f"Reorganized Output Base Dir: {args.output_base_dir.resolve()}")
    script_logger.info(f"Output Index CSV: {args.output_index_csv.resolve()}")
    script_logger.info(f"Key file for status check: '{args.key_file_check}'")

    # 1. Validate Inputs
    if not args.mapping_csv.is_file():
        script_logger.error(f"Mapping CSV file not found: {args.mapping_csv}")
        sys.exit(1)
    if not args.output_base_dir.is_dir():
        script_logger.error(f"Reorganized output base directory not found or not a directory: {args.output_base_dir}")
        sys.exit(1)
    try: # Check if output directory is writable
        args.output_index_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_index_csv.parent / f".write_test_{os.getpid()}", "w") as f:
            f.write("test")
        os.remove(args.output_index_csv.parent / f".write_test_{os.getpid()}")
    except Exception as e:
        script_logger.error(f"Cannot write to output directory '{args.output_index_csv.parent}': {e}")
        sys.exit(1)

    # 2. Load Mappings
    mappings = parse_csv_mapping(str(args.mapping_csv))
    if not mappings:
        script_logger.error("No mappings loaded from the CSV file. Cannot generate index.")
        sys.exit(1)

    # 3. Process each mapping entry
    index_data = []
    script_logger.info(f"Checking status for {len(mappings)} documents based on mapping...")
    for mapping_entry in tqdm.tqdm(mappings, desc="Generating Index", unit="doc"):
        old_id = mapping_entry['old_id']
        new_id = mapping_entry['new_id']

        # Derive corpus prefix (consistent with main script)
        corpus_prefix_match = re.match(r'^([a-zA-Z0-9\-_]+?)_', old_id)
        corpus_prefix = corpus_prefix_match.group(1) if corpus_prefix_match else "unknown_corpus"

        # Get all expected paths for this document
        expected_paths = get_expected_output_paths(
            str(args.output_base_dir), corpus_prefix, new_id
        )

        # Check for existence of the key file to determine status
        key_file_path = expected_paths.get(args.key_file_check)
        status = "unknown_key"
        is_complete = False
        if key_file_path:
            if os.path.exists(key_file_path):
                status = "complete"
                is_complete = True
            else:
                status = "incomplete"
        else:
            script_logger.warning(f"Key file '{args.key_file_check}' definition not found in expected paths for {old_id}->{new_id}. Status set to 'unknown_key'.")


        # Create row data including all paths
        row_data = {
            'old_id': old_id,
            'new_id': new_id,
            'corpus_prefix': corpus_prefix,
            'reorganization_status': status,
            'is_complete': is_complete,
            **expected_paths # Add all path columns directly
        }
        index_data.append(row_data)

    # 4. Create DataFrame
    if not index_data:
        script_logger.warning("No index data was generated (perhaps the mapping was empty after filtering?).")
        master_df = pd.DataFrame() # Create empty DF if no data
    else:
        master_df = pd.DataFrame(index_data)

        # Define desired column order (optional, but nice)
        ordered_cols = [
            'old_id', 'new_id', 'corpus_prefix', 'reorganization_status', 'is_complete',
            'output_dir', 'texts_dir', 'annotations_dir',
            'output_conllu', 'output_txt_joined', 'output_txt_fullstop',
            'output_csv_lemma', 'output_csv_upos', 'output_csv_stop',
            'output_csv_dot', 'output_csv_ner', 'output_ner_tags',
            'source_original_txt',
            'ner_mismatch_info', 'ner_stats_file', 'ner_error_info'
        ]
        # Ensure all expected columns are present, even if they were somehow missed
        # (e.g., if get_expected_output_paths changed)
        final_cols = [col for col in ordered_cols if col in master_df.columns]
        # Add any extra columns that might have been added unexpectedly
        extra_cols = [col for col in master_df.columns if col not in final_cols]
        master_df = master_df[final_cols + extra_cols]


    # 5. Save the master index
    try:
        script_logger.info(f"Saving index with {len(master_df)} rows to: {args.output_index_csv}")
        master_df.to_csv(args.output_index_csv, index=False, quoting=csv.QUOTE_ALL) # Use QUOTE_ALL for safety with paths
        script_logger.info("Master reorganization index saved successfully.")

        # Print status summary
        if not master_df.empty:
            script_logger.info(f"Status Summary:\n{master_df['reorganization_status'].value_counts().to_string()}")
        else:
            script_logger.info("Status Summary: No documents indexed.")

    except Exception as e:
        script_logger.exception(f"Failed to save master index to {args.output_index_csv}.")
        sys.exit(1)

    script_logger.info("--- Master Reorganization Index Generation Complete ---")


if __name__ == "__main__":
    main()
