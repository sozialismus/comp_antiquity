import argparse
import logging
import sys
import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional

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

# --- Helper function to define the output structure ---
def get_expected_output_paths(output_base_dir: str, corpus_prefix: str, new_id: str) -> Dict[str, str]:
    """Calculates the expected standard output paths for a reorganized document."""
    abs_output_base_dir = os.path.abspath(output_base_dir)
    output_dir = os.path.join(abs_output_base_dir, corpus_prefix, new_id)
    texts_dir = os.path.join(output_dir, "texts")
    annotations_dir = os.path.join(output_dir, "annotations")
    return {
        "output_dir": output_dir, "texts_dir": texts_dir, "annotations_dir": annotations_dir,
        "output_txt_joined": os.path.join(texts_dir, f"{new_id}-joined.txt"),
        "output_txt_fullstop": os.path.join(texts_dir, f"{new_id}-fullstop.txt"),
        "output_csv_lemma": os.path.join(annotations_dir, f"{new_id}-lemma.csv"),
        "output_csv_upos": os.path.join(annotations_dir, f"{new_id}-upos.csv"),
        "output_csv_stop": os.path.join(annotations_dir, f"{new_id}-stop.csv"),
        "output_csv_dot": os.path.join(annotations_dir, f"{new_id}-dot.csv"),
        "output_csv_ner": os.path.join(annotations_dir, f"{new_id}-ner.csv"),
        "output_conllu": os.path.join(annotations_dir, f"{new_id}-conllu.conllu"),
        "output_ner_tags": os.path.join(annotations_dir, f"{new_id}-ner_tags.txt"),
        "source_original_txt": os.path.join(texts_dir, f"{new_id}-original.txt"),
        "ner_mismatch_info": os.path.join(annotations_dir, f"{new_id}_ner_mismatch_info.json"),
        "ner_stats_file": os.path.join(annotations_dir, f"{new_id}_ner_stats.json"),
        "ner_error_info": os.path.join(annotations_dir, f"{new_id}_ner_error_info.json"),
    }

# --- Helper function to parse the mapping file, enforcing 1-to-1 mapping ---
def parse_csv_mapping(csv_path: str) -> List[Dict[str, str]]:
    """
    Parses mapping CSV, enforcing a unique one-to-one mapping between old_id and new_id.
    """
    mappings_list, processed_old_ids, processed_new_ids = [], set(), set()
    abs_csv_path = os.path.abspath(csv_path)
    script_logger.info(f"Loading and validating mapping file: {abs_csv_path}")
    try:
        df = pd.read_csv(abs_csv_path, dtype={'document_id': str, 'sort_id': str}, low_memory=False)
        id_col, sort_col = 'document_id', 'sort_id'
        if id_col not in df.columns or sort_col not in df.columns:
            script_logger.error(f"Mapping file missing '{id_col}' or '{sort_col}' columns.")
            return []

        old_id_dupes, new_id_dupes, missing_ids = 0, 0, 0
        for index, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Parsing mapping CSV", unit="row"):
            old_id = str(row.get(id_col, '')).strip()
            new_id = str(row.get(sort_col, '')).strip()
            
            if not old_id: continue
            if not new_id:
                script_logger.debug(f"Skipping row {index+2}: Missing sort_id for doc '{old_id}'.")
                missing_ids += 1
                continue
            
            if old_id in processed_old_ids:
                old_id_dupes += 1
                continue
            if new_id in processed_new_ids:
                script_logger.warning(f"Duplicate sort_id '{new_id}' for doc '{old_id}' (row {index+2}). Skipping to enforce 1-to-1 mapping.")
                new_id_dupes += 1
                continue
                
            mappings_list.append({'old_id': old_id, 'new_id': new_id})
            processed_old_ids.add(old_id)
            processed_new_ids.add(new_id)

        if old_id_dupes > 0: script_logger.warning(f"Skipped {old_id_dupes} duplicate document_id entries.")
        if new_id_dupes > 0: script_logger.warning(f"Skipped {new_id_dupes} duplicate sort_id entries.")
        if missing_ids > 0: script_logger.warning(f"Skipped {missing_ids} entries with missing sort_id.")
        script_logger.info(f"Successfully loaded {len(mappings_list)} unique one-to-one mappings.")
    except Exception as e:
        script_logger.error(f"Error parsing mapping file '{abs_csv_path}': {e}", exc_info=True)
        return []
    return mappings_list

# --- Helper functions for the cross-referencing feature ---

def find_source_txt_path(base_dir: str, old_id: str, corpus_prefix: str) -> Optional[str]:
    """Tries to find the original source .txt file using common patterns."""
    base_path = Path(base_dir).resolve()
    # Define search patterns in order of likelihood
    patterns = []
    if corpus_prefix != "unknown_corpus":
        filename_part = old_id.replace(f"{corpus_prefix}_", "", 1)
        patterns.append(base_path / corpus_prefix / f"{filename_part}.txt")
        patterns.append(base_path / corpus_prefix / f"{old_id}.txt")
    patterns.append(base_path / f"{old_id}.txt")
    
    for path in patterns:
        if path.is_file():
            return str(path)
    return None

def get_file_snippet(file_path: Optional[str], num_chars: int = 100) -> str:
    """Reads the first N characters from a file, cleaning for CSV output."""
    if not file_path or not os.path.exists(file_path): return ""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read(num_chars).replace('\n', ' ').replace('\r', '').strip()
    except Exception: return ""

def get_conllu_text_snippet(file_path: Optional[str], num_chars: int = 100) -> str:
    """Extracts and concatenates text from '# text =' lines in a CoNLL-U file."""
    if not file_path or not os.path.exists(file_path): return ""
    full_text = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("# text = "):
                    full_text.append(line[9:].strip())
        reconstructed_text = " ".join(full_text)
        return reconstructed_text[:num_chars].strip()
    except Exception: return ""

def normalize_for_comparison(text: str) -> str:
    """Prepares a string for comparison by lowercasing and removing non-alphanumerics."""
    return re.sub(r'[^a-z0-9]', '', text.lower())

# --- Argument Parser Setup ---
def create_parser() -> argparse.ArgumentParser:
    """Defines command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Generates a master index of reorganized corpus files, with content cross-referencing.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mapping-csv", type=Path, required=True, help="Path to the CSV mapping 'document_id' to 'sort_id'.")
    parser.add_argument("--output-base-dir", type=Path, required=True, help="Path to the base directory WHERE reorganized files were saved.")
    parser.add_argument("--source-base-dir", type=Path, required=True, help="Path to the base directory of the ORIGINAL source .txt files for cross-referencing.")
    parser.add_argument("--output-index-csv", type=Path, required=True, help="Path for the output master index CSV file.")
    parser.add_argument("--key-file-check", type=str, default="output_conllu", choices=list(get_expected_output_paths("b", "p", "i").keys()), help="Which file's existence determines 'complete' status.")
    return parser

# --- Main script execution logic ---
def main():
    parser = create_parser()
    args = parser.parse_args()

    script_logger.info("--- Generating Reorganization Master Index (with Cross-Referencing) ---")
    script_logger.info(f"Mapping CSV: {args.mapping_csv.resolve()}")
    script_logger.info(f"Reorganized Output Base Dir: {args.output_base_dir.resolve()}")
    script_logger.info(f"Original Source Base Dir: {args.source_base_dir.resolve()}")
    script_logger.info(f"Output Index CSV: {args.output_index_csv.resolve()}")

    if not all([args.mapping_csv.is_file(), args.output_base_dir.is_dir(), args.source_base_dir.is_dir()]):
        script_logger.error("One or more input paths are invalid. Please check file/directory existence."); sys.exit(1)
    
    try:
        args.output_index_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_index_csv.parent / f".write_test_{os.getpid()}", "w") as f: f.write("test")
        os.remove(args.output_index_csv.parent / f".write_test_{os.getpid()}")
    except Exception as e:
        script_logger.error(f"Cannot write to output directory '{args.output_index_csv.parent}': {e}"); sys.exit(1)

    mappings = parse_csv_mapping(str(args.mapping_csv))
    if not mappings: script_logger.error("No valid 1-to-1 mappings loaded. Exiting."); sys.exit(1)

    index_data = []
    script_logger.info(f"Checking status and cross-referencing content for {len(mappings)} documents...")
    
    for mapping_entry in tqdm.tqdm(mappings, desc="Generating Index", unit="doc"):
        old_id, new_id = mapping_entry['old_id'], mapping_entry['new_id']
        corpus_prefix_match = re.match(r'^([a-zA-Z0-9\-_]+?)_', old_id)
        corpus_prefix = corpus_prefix_match.group(1) if corpus_prefix_match else "unknown_corpus"
        
        expected_paths = get_expected_output_paths(str(args.output_base_dir), corpus_prefix, new_id)
        key_file_path = expected_paths.get(args.key_file_check)
        status = "complete" if key_file_path and os.path.exists(key_file_path) else "incomplete"
        
        # Cross-referencing logic
        cross_ref_status, source_snippet, conllu_snippet = "not_checked", "", ""
        source_txt_path = find_source_txt_path(str(args.source_base_dir), old_id, corpus_prefix)
        
        if status == "incomplete":
            cross_ref_status = "skipped_incomplete"
        elif not source_txt_path:
            cross_ref_status = "error_source_missing"
        else:
            source_snippet = get_file_snippet(source_txt_path)
            conllu_snippet = get_conllu_text_snippet(expected_paths.get("output_conllu"))
            if not source_snippet: cross_ref_status = "error_source_empty"
            elif not conllu_snippet: cross_ref_status = "error_conllu_empty"
            else:
                norm_source = normalize_for_comparison(source_snippet)
                norm_conllu = normalize_for_comparison(conllu_snippet)
                cross_ref_status = "match" if norm_source == norm_conllu else "mismatch"
        
        index_data.append({
            'old_id': old_id, 'new_id': new_id, 'corpus_prefix': corpus_prefix,
            'reorganization_status': status, 'cross_ref_status': cross_ref_status,
            'source_snippet': source_snippet, 'conllu_snippet': conllu_snippet,
            **expected_paths
        })

    if not index_data: script_logger.warning("No index data was generated."); sys.exit(0)
    master_df = pd.DataFrame(index_data)

    # Define the desired column order for better readability
    ordered_cols = [
        'old_id', 'new_id', 'corpus_prefix', 'reorganization_status',
        'cross_ref_status', 'source_snippet', 'conllu_snippet',
        'output_dir', 'texts_dir', 'annotations_dir', 'output_conllu'
    ]
    final_cols = [c for c in ordered_cols if c in master_df.columns]
    extra_cols = [c for c in master_df.columns if c not in final_cols]
    master_df = master_df[final_cols + extra_cols]

    try:
        script_logger.info(f"Saving index with {len(master_df)} rows to: {args.output_index_csv}")
        master_df.to_csv(args.output_index_csv, index=False, quoting=csv.QUOTE_ALL)
        script_logger.info("Master reorganization index saved successfully.")
        script_logger.info(f"Reorganization Status Summary:\n{master_df['reorganization_status'].value_counts().to_string()}")
        script_logger.info(f"Cross-Reference Status Summary:\n{master_df['cross_ref_status'].value_counts().to_string()}")
    except Exception as e:
        script_logger.exception(f"Failed to save master index to {args.output_index_csv}."); sys.exit(1)

    script_logger.info("--- Master Reorganization Index Generation Complete ---")

if __name__ == "__main__":
    main()
