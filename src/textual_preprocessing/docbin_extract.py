#!/usr/bin/env python3
"""
Script to reorganize corpus documents according to the database schema.
Reads document mappings from CSV and uses indices to locate docbin files.
Extracts various formats (TXT, CSVs, CoNLL-U) using specified spaCy models
in separate conda environments via temporary scripts. Includes NER tags in CoNLL-U MISC column.
Includes comprehensive logging via wandb and CSV.

CSV Delimiter Confirmation: All CSV files read or written by this script
use COMMAS (,) as delimiters. This includes the input mapping/index CSVs
and all generated CSV annotation files.
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
import tempfile # For temporary file handling
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import tqdm # Keep tqdm for progress bars
import wandb


# --- Utility function to run commands via temp script ---
# This centralizes the logic for creating, running, and cleaning up temp scripts
def run_python_script_in_conda_env(
    conda_env: str,
    script_content: str,
    log_context: Dict[str, Any], # Pass logger args like old_id, new_id etc.
    logger: Optional['FileOperationLogger'] = None,
    timeout: int = 300 # Default timeout in seconds
) -> bool:
    """
    Writes python code to a temp file and runs it in a specified conda env.

    Returns:
        True on success, False on failure.
    """
    temp_script_fd, temp_script_path = tempfile.mkstemp(suffix='.py', text=True)
    os.close(temp_script_fd)
    success = False
    log_file_type = log_context.get('file_type', 'unknown_script') # Get file type for logging

    try:
        with open(temp_script_path, 'w', encoding='utf-8') as f_script:
            f_script.write(script_content)

        cmd = f"conda run -n {conda_env} python {temp_script_path}"
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=timeout)

        # Check stderr for warnings/errors printed by the script itself (e.g., print to sys.stderr)
        if result.stderr and logger:
             logger.log_operation(
                 **log_context, # Unpack context like old_id, new_id, source, dest etc.
                 operation_type="extract", file_type=log_file_type, status="warning",
                 details=f"Extraction script stderr: {result.stderr.strip()}"
             )

        if logger:
            logger.log_operation(
                **log_context,
                operation_type="extract", file_type=log_file_type, status="success"
            )
        success = True

    except subprocess.TimeoutExpired as e:
         error_details = f"Command timed out ({e.timeout}s): {e.cmd}\nStderr: {e.stderr}\nStdout: {e.stdout}"
         if logger:
             logger.log_operation(
                 **log_context, operation_type="extract", file_type=log_file_type, status="failed",
                 details=error_details
             )
    except subprocess.CalledProcessError as e:
        error_details = f"Command failed: {e.cmd}\nStderr: {e.stderr}\nStdout: {e.stdout}"
        if logger:
            logger.log_operation(
                 **log_context, operation_type="extract", file_type=log_file_type, status="failed",
                 details=error_details
            )
    except Exception as e:
        if logger:
            logger.log_operation(
                 **log_context, operation_type="extract", file_type=log_file_type, status="failed",
                 details=f"Unexpected error setting up/running temp script: {str(e)}"
            )
    finally:
        # Clean up temp script
        if os.path.exists(temp_script_path):
            try:
                os.unlink(temp_script_path)
            except OSError as unlink_e:
                 if logger: logger.log_operation(
                     **log_context, source_file=temp_script_path, destination_file="",
                     operation_type="cleanup", file_type="temp_script", status="failed",
                     details=f"Could not remove temp script: {unlink_e}"
                 )
    return success

class FileOperationLogger:
    """
    Logger to track file operations during the document extraction process.
    Logs to CSV and optionally to Weights & Biases.
    """
    def __init__(self, log_file_path: str, use_wandb: bool = True, wandb_project: str = "corpus-reorganization"):
        self.log_file_path = log_file_path
        self.log_entries = []
        self.use_wandb = use_wandb
        self.run_name = f"reorganization-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        try:
            # Create log file with headers (using comma delimiter)
            with open(log_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile) # Defaults to comma
                writer.writerow([
                    'timestamp', 'old_id', 'new_id', 'corpus_prefix',
                    'source_file', 'destination_file', 'operation_type',
                    'file_type', 'status', 'details'
                ])
        except OSError as e:
            print(f"!!! Error creating log file {log_file_path}: {e}. Logging disabled for CSV.")
            self.log_file_path = None # Prevent further attempts

        # Initialize wandb if enabled
        if use_wandb:
            try:
                wandb.init(project=wandb_project, name=self.run_name)
                wandb.config.update({
                    "log_file": log_file_path if self.log_file_path else "N/A",
                    "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            except Exception as e:
                print(f"!!! Error initializing wandb: {e}. Wandb logging disabled.")
                self.use_wandb = False # Disable wandb if init fails

    def log_operation(
        self,
        old_id: str,
        new_id: str,
        corpus_prefix: str,
        source_file: str,
        destination_file: str,
        operation_type: str,
        file_type: str,
        status: str = "success",
        details: str = ""
    ) -> None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        src_str = str(source_file)
        dest_str = str(destination_file)
        details_str = str(details) # Ensure details are string

        entry = {
            'timestamp': timestamp,
            'old_id': old_id,
            'new_id': new_id,
            'corpus_prefix': corpus_prefix,
            'source_file': src_str,
            'destination_file': dest_str,
            'operation_type': operation_type,
            'file_type': file_type,
            'status': status,
            'details': details_str
        }
        self.log_entries.append(entry)

        # Write to CSV immediately (if path is valid)
        if self.log_file_path:
            try:
                with open(self.log_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile) # Defaults to comma
                    writer.writerow([
                        timestamp, old_id, new_id, corpus_prefix,
                        src_str, dest_str, operation_type,
                        file_type, status, details_str
                    ])
            except OSError as e:
                 print(f"!!! Warning: Failed to write to CSV log {self.log_file_path}: {e}")

        # Log to wandb if enabled
        if self.use_wandb:
            try:
                # Log individual operations with a prefix for clarity
                log_payload = {f"op_{operation_type}/{file_type}_{status}": 1} # Count occurrences
                # Add more detailed log entry keyed by document ID and timestamp?
                # Using wandb.log steps might be better for sequential events.
                wandb.log({
                    "file_operation_details": entry, # Log full details
                    f"status_counts/{status}": 1,    # Increment status count
                    f"op_counts/{operation_type}": 1, # Increment op count
                    f"type_counts/{file_type}": 1,   # Increment file type count
                    # Custom charts might need step= or commit=False/True logic
                })
            except Exception as e:
                 # Prevent wandb errors from crashing the script
                 print(f"!!! Warning: Failed to log operation to wandb: {e}")

    def summarize_and_close(self) -> Dict[str, Any]:
        # --- Calculation logic remains the same ---
        total_operations = len(self.log_entries)
        successful_operations = sum(1 for entry in self.log_entries if entry['status'] == 'success')
        failed_operations = sum(1 for entry in self.log_entries if entry['status'] == 'failed')
        warning_operations = sum(1 for entry in self.log_entries if entry['status'] == 'warning')

        operation_counts = {}
        for entry in self.log_entries:
            op_type = entry['operation_type']
            operation_counts[op_type] = operation_counts.get(op_type, 0) + 1

        file_type_counts = {}
        for entry in self.log_entries:
            file_type = entry['file_type']
            file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1

        unique_old_ids = set(entry['old_id'] for entry in self.log_entries if entry['operation_type'] == 'process_start')
        unique_new_ids = set(entry['new_id'] for entry in self.log_entries if entry['operation_type'] == 'process_start')

        summary = {
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'failed_operations': failed_operations,
            'warning_operations': warning_operations,
            'operation_counts': operation_counts,
            'file_type_counts': file_type_counts,
            'unique_documents_processed': len(unique_old_ids),
            'unique_new_ids_created': len(unique_new_ids),
            'success_rate': successful_operations / total_operations if total_operations > 0 else 0,
        }

        # Log summary to wandb
        if self.use_wandb:
            try:
                # Log final summary metrics
                wandb.summary.update(summary) # Use wandb.summary for final values
                wandb.summary.update({
                    "completion_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

                # Create and log summary table
                columns = ["Metric", "Value"]
                data = [
                    ["Total Operations", total_operations],
                    ["Successful Operations", successful_operations],
                    ["Failed Operations", failed_operations],
                    ["Warning Operations", warning_operations],
                    ["Success Rate", f"{summary['success_rate']:.2%}"],
                    ["Unique Documents Processed", len(unique_old_ids)],
                    ["Unique New IDs Created", len(unique_new_ids)]
                ]
                for op_type, count in operation_counts.items():
                    data.append([f"Operation: {op_type}", count])
                for file_type, count in file_type_counts.items():
                    data.append([f"File Type: {file_type}", count])

                summary_table = wandb.Table(columns=columns, data=data)
                wandb.log({"summary_statistics_table": summary_table})

                wandb.finish()
            except Exception as e:
                 print(f"!!! Warning: Failed to log summary/finish wandb run: {e}")

        return summary


# --- Extraction Functions (Now using run_python_script_in_conda_env) ---

def extract_text_from_docbin(
    conda_env: str, docbin_path: str, output_txt_path: str, logger: Optional[FileOperationLogger] = None, **log_ctx
) -> bool:
    """Extract plain text using temporary script."""
    script_content = f"""
import sys, spacy, os
from spacy.tokens import DocBin
try:
    nlp = spacy.load('grc_proiel_trf')
    doc_bin = DocBin().from_disk(r'{docbin_path}')
    # Handle potential errors during DocBin loading or if it's empty
    docs = list(doc_bin.get_docs(nlp.vocab))
    if not docs:
        raise ValueError("DocBin file contains no documents.")
    doc = docs[0]
    os.makedirs(os.path.dirname(r'{output_txt_path}'), exist_ok=True) # Ensure dir exists
    with open(r'{output_txt_path}', 'w', encoding='utf-8') as f:
        f.write(doc.text)
except Exception as e:
    print(f"Error in extract_text_from_docbin script: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
    log_ctx.update({'source_file': docbin_path, 'destination_file': output_txt_path, 'file_type': 'txt'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger)

def create_fullstop_file(
    input_txt_path: str, output_txt_path: str, logger: Optional[FileOperationLogger] = None, **log_ctx
) -> bool:
    """Create a fullstop file. (Runs in current env, no conda needed)"""
    log_ctx.update({'source_file': input_txt_path, 'destination_file': output_txt_path, 'file_type': 'txt-fullstop', 'operation_type': 'create'})
    try:
        # Check if input file exists before reading
        if not os.path.exists(input_txt_path):
             if logger: logger.log_operation(**log_ctx, status="skipped", details="Input text file missing")
             return False # Indicate failure if input is missing

        with open(input_txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        text = re.sub(r'\.(?!\.)', '.\n', text)
        text = re.sub(r'\s+\n', '\n', text)
        text = re.sub(r'\n\s+', '\n', text).strip()

        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True) # Ensure dir exists
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        if logger: logger.log_operation(**log_ctx, status="success")
        return True
    except Exception as e:
        if logger: logger.log_operation(**log_ctx, status="failed", details=str(e))
        return False

def extract_lemma_csv(
    conda_env: str, docbin_path: str, output_csv_path: str, logger: Optional[FileOperationLogger] = None, **log_ctx
) -> bool:
    """Extract lemma CSV using temporary script."""
    script_content = f"""
import sys, csv, spacy, os
from spacy.tokens import DocBin
try:
    nlp = spacy.load('grc_proiel_trf')
    doc_bin = DocBin().from_disk(r'{docbin_path}')
    docs = list(doc_bin.get_docs(nlp.vocab))
    if not docs: raise ValueError("DocBin file contains no documents.")
    doc = docs[0]
    os.makedirs(os.path.dirname(r'{output_csv_path}'), exist_ok=True)
    with open(r'{output_csv_path}', 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile) # Comma delimited
        writer.writerow(['ID', 'TOKEN', 'LEMMA'])
        count = 0
        for token in doc:
            count += 1
            writer.writerow([count, token.text, token.lemma_])
except Exception as e:
    print(f"Error in extract_lemma_csv script: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
    log_ctx.update({'source_file': docbin_path, 'destination_file': output_csv_path, 'file_type': 'csv-lemma'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger)

def extract_upos_csv(
    conda_env: str, docbin_path: str, output_csv_path: str, logger: Optional[FileOperationLogger] = None, **log_ctx
) -> bool:
    """Extract UPOS CSV using temporary script."""
    script_content = f"""
import sys, csv, spacy, os
from spacy.tokens import DocBin
try:
    nlp = spacy.load('grc_proiel_trf')
    doc_bin = DocBin().from_disk(r'{docbin_path}')
    docs = list(doc_bin.get_docs(nlp.vocab))
    if not docs: raise ValueError("DocBin file contains no documents.")
    doc = docs[0]
    os.makedirs(os.path.dirname(r'{output_csv_path}'), exist_ok=True)
    with open(r'{output_csv_path}', 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'TOKEN', 'UPOS'])
        count = 0
        for token in doc:
            count += 1
            writer.writerow([count, token.text, token.pos_])
except Exception as e:
    print(f"Error in extract_upos_csv script: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
    log_ctx.update({'source_file': docbin_path, 'destination_file': output_csv_path, 'file_type': 'csv-upos'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger)

def extract_stop_csv(
    conda_env: str, docbin_path: str, output_csv_path: str, logger: Optional[FileOperationLogger] = None, **log_ctx
) -> bool:
    """Extract stopword CSV using temporary script."""
    script_content = f"""
import sys, csv, spacy, os
from spacy.tokens import DocBin
try:
    nlp = spacy.load('grc_proiel_trf')
    doc_bin = DocBin().from_disk(r'{docbin_path}')
    docs = list(doc_bin.get_docs(nlp.vocab))
    if not docs: raise ValueError("DocBin file contains no documents.")
    doc = docs[0]
    os.makedirs(os.path.dirname(r'{output_csv_path}'), exist_ok=True)
    with open(r'{output_csv_path}', 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'TOKEN', 'IS_STOP'])
        count = 0
        for token in doc:
            count += 1
            writer.writerow([count, token.text, 'TRUE' if token.is_stop else 'FALSE'])
except Exception as e:
    print(f"Error in extract_stop_csv script: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
    log_ctx.update({'source_file': docbin_path, 'destination_file': output_csv_path, 'file_type': 'csv-stop'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger)

def extract_ner_csv(
    conda_env: str, docbin_path: str, output_csv_path: str, logger: Optional[FileOperationLogger] = None, **log_ctx
) -> bool:
    """Extract NER CSV using temporary script."""
    script_content = f"""
import sys, csv, spacy, os
from spacy.tokens import DocBin
try:
    nlp = spacy.load('grc_ner_trf') # Use NER model
    doc_bin = DocBin().from_disk(r'{docbin_path}')
    docs = list(doc_bin.get_docs(nlp.vocab))
    if not docs: raise ValueError("DocBin file contains no documents.")
    doc = docs[0]
    os.makedirs(os.path.dirname(r'{output_csv_path}'), exist_ok=True)
    with open(r'{output_csv_path}', 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'TOKEN', 'NER'])
        count = 0
        for token in doc:
            count += 1
            writer.writerow([count, token.text, token.ent_type_ if token.ent_type_ else 'O'])
except Exception as e:
    print(f"Error in extract_ner_csv script: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
    log_ctx.update({'source_file': docbin_path, 'destination_file': output_csv_path, 'file_type': 'csv-ner'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger)

def extract_ner_tags_to_file(
    conda_env: str, ner_docbin_path: str, output_tags_path: str, logger: Optional[FileOperationLogger] = None, **log_ctx
) -> bool:
    """Extract only NER tags using temporary script."""
    script_content = f"""
import sys, spacy, os
from spacy.tokens import DocBin
try:
    nlp = spacy.load('grc_ner_trf') # Use NER model
    doc_bin = DocBin().from_disk(r'{ner_docbin_path}')
    docs = list(doc_bin.get_docs(nlp.vocab))
    if not docs: raise ValueError("DocBin file contains no documents.")
    doc = docs[0]
    os.makedirs(os.path.dirname(r'{output_tags_path}'), exist_ok=True)
    with open(r'{output_tags_path}', 'w', encoding='utf-8') as f:
        f.write('\\n'.join([token.ent_type_ if token.ent_type_ else 'O' for token in doc]))
except Exception as e:
    print(f"Error in extract_ner_tags_to_file script: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
    log_ctx.update({'source_file': ner_docbin_path, 'destination_file': output_tags_path, 'file_type': 'ner-tags'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger)

def extract_dot_csv(
    conda_env: str, docbin_path: str, output_csv_path: str, logger: Optional[FileOperationLogger] = None, **log_ctx
) -> bool:
    """Extract punctuation CSV using temporary script."""
    script_content = f"""
import sys, csv, spacy, os
from spacy.tokens import DocBin
try:
    nlp = spacy.load('grc_proiel_trf')
    doc_bin = DocBin().from_disk(r'{docbin_path}')
    docs = list(doc_bin.get_docs(nlp.vocab))
    if not docs: raise ValueError("DocBin file contains no documents.")
    doc = docs[0]
    os.makedirs(os.path.dirname(r'{output_csv_path}'), exist_ok=True)
    with open(r'{output_csv_path}', 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'TOKEN', 'IS_PUNCT'])
        count = 0
        for token in doc:
            count += 1
            writer.writerow([count, token.text, 'TRUE' if token.is_punct else 'FALSE'])
except Exception as e:
    print(f"Error in extract_dot_csv script: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
    log_ctx.update({'source_file': docbin_path, 'destination_file': output_csv_path, 'file_type': 'csv-dot'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger)


def extract_conllu_file(
    conda_env: str, main_docbin_path: str, output_conllu_path: str, doc_id: str,
    ner_tags_path: Optional[str], logger: Optional[FileOperationLogger] = None, **log_ctx
) -> bool:
    """Extract CoNLL-U using temporary script, incorporating NER tags."""
    # FIX for f-string backslash issue: Construct the # text line separately.
    script_content = f"""
# -*- coding: utf-8 -*-
# Note: Ensure target environment's Python handles UTF-8 correctly.
import sys
import spacy
import os
from spacy.tokens import DocBin

# --- Configuration ---
main_docbin_path = r'{main_docbin_path}'
output_conllu_path = r'{output_conllu_path}'
ner_tags_path = {repr(ner_tags_path)}
doc_id_str = '{doc_id}'
# --- End Configuration ---

try:
    # Load main linguistic model (proiel)
    nlp = spacy.load('grc_proiel_trf')

    # Load DocBin from main model
    doc_bin = DocBin().from_disk(main_docbin_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    if not docs: raise ValueError("DocBin file contains no documents.")
    doc = docs[0]

    # Load NER tags if path is provided
    ner_tags = None
    if ner_tags_path and os.path.exists(ner_tags_path):
        try:
            with open(ner_tags_path, 'r', encoding='utf-8') as f_ner:
                ner_tags = [line.strip() for line in f_ner if line.strip()]
            if len(ner_tags) != len(doc):
                print(f"Warning (doc {{doc_id_str}}): Mismatch token count ({{len(doc)}}) vs NER tag count ({{len(ner_tags)}}).", file=sys.stderr)
                # ner_tags = None # Optionally disable NER on mismatch
        except Exception as e:
            print(f"Warning (doc {{doc_id_str}}): Could not read NER tags from {{ner_tags_path}}: {{e}}.", file=sys.stderr)
            ner_tags = None
    elif ner_tags_path:
         print(f"Warning (doc {{doc_id_str}}): NER tags path does not exist: {{ner_tags_path}}.", file=sys.stderr)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_conllu_path), exist_ok=True)

    # Extract CoNLL-U format
    with open(output_conllu_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"# newdoc id = {{doc_id_str}}\\n")

        sent_id_counter = 1
        for sent in doc.sents:
            # Write sentence header
            sent_text_clean = sent.text.replace('\\n', ' ').replace('\\r', '') # Clean sentence text
            f_out.write(f"# sent_id = {{doc_id_str}}-{{sent_id_counter}}\\n")
            f_out.write(f"# text = {{sent_text_clean}}\\n") # Write cleaned text

            token_sent_id_counter = 1
            for token in sent:
                head_id = token.head.i - sent.start + 1 if token.head.i != token.i else 0
                feats = str(token.morph) if token.morph else "_"
                if not feats: feats = "_"

                misc_parts = []
                if ner_tags:
                    try:
                        ner_tag = ner_tags[token.i]
                        if ner_tag and ner_tag != 'O': # Only add non-O tags by default
                             misc_parts.append(f"NER={{ner_tag}}")
                    except IndexError:
                         misc_parts.append("NER=Error") # Mismatch detected earlier

                if token.i + 1 < len(doc) and doc[token.i+1].idx == token.idx + len(token.text):
                     misc_parts.append("SpaceAfter=No")

                misc_field = "|".join(misc_parts) if misc_parts else "_"

                line = (f"{{token_sent_id_counter}}\\t{{token.text}}\\t{{token.lemma_}}\\t{{token.pos_}}\\t"
                       f"{{token.tag_}}\\t{{feats}}\\t{{head_id}}\\t"
                       f"{{token.dep_}}\\t_\\t{{misc_field}}\\n")
                f_out.write(line)
                token_sent_id_counter += 1

            f_out.write("\\n")
            sent_id_counter += 1

except Exception as e:
    print(f"Error in extract_conllu_file script for doc {{doc_id_str}}: {{e}}", file=sys.stderr)
    # Print traceback for more details
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

"""
    log_ctx.update({'source_file': main_docbin_path, 'destination_file': output_conllu_path, 'file_type': 'conllu'})
    # CoNLL-U can take longer, increase timeout maybe
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger, timeout=600)


# --- Helper Functions (load_index updated for 'processed_path') ---

def load_index(index_path: str) -> Dict[str, str]:
    """
    Load an index file (CSV, comma-delimited) that maps document IDs to file paths.
    Expects columns 'document_id' and 'processed_path'.
    """
    index = {}
    try:
        df = pd.read_csv(index_path) # Defaults to comma delimiter
        required_path_col = 'processed_path'
        if 'document_id' in df.columns and required_path_col in df.columns:
            for _, row in df.iterrows():
                doc_id = str(row['document_id']).strip()
                file_path = str(row[required_path_col]).strip()
                if doc_id and file_path: # Ensure neither are empty strings
                     index[doc_id] = file_path
                else:
                     print(f"Warning: Skipping row in '{index_path}' with empty document_id or {required_path_col}.")
        else:
            print(f"Warning: Index file '{index_path}' missing required columns 'document_id' or '{required_path_col}'.")
    except FileNotFoundError:
         print(f"Error: Index file not found at '{index_path}'")
         # Depending on severity, you might want to raise the exception or exit
         # raise SystemExit(f"Exiting: Index file not found: {index_path}")
    except Exception as e:
         print(f"Error loading index file '{index_path}': {e}")
         # raise
    return index

def parse_csv_mapping(csv_path: str) -> Dict[str, str]:
    """
    Parse the CSV file (comma-delimited) containing old to new ID mappings.
    Returns a dictionary mapping old IDs (str) to new numeric IDs (str).
    """
    mappings = {}
    try:
        df = pd.read_csv(csv_path) # Defaults to comma delimiter
        if 'document_id' in df.columns and 'sort_id' in df.columns:
             for index, row in df.iterrows():
                old_id = str(row['document_id']).strip()
                new_id_val = row['sort_id']
                # Check for NaN/NaT explicitly before converting
                if pd.notna(new_id_val):
                    new_id = str(int(new_id_val))
                else:
                    new_id = None

                if old_id and new_id:
                    mappings[old_id] = new_id
                elif old_id:
                    print(f"Warning: Missing or invalid sort_id for document_id '{old_id}' in '{csv_path}'. Skipping.")
        else:
             print(f"Warning: Mapping file '{csv_path}' missing required columns 'document_id' or 'sort_id'.")
    except FileNotFoundError:
         print(f"Error: Mapping file not found at '{csv_path}'")
         # raise SystemExit(f"Exiting: Mapping file not found: {csv_path}")
    except Exception as e:
         print(f"Error parsing mapping file '{csv_path}': {e}")
         # raise
    return mappings


def find_original_text(doc_id: str, base_dir: str) -> Optional[str]:
    """
    Try to locate the original text file for a document ID.
    Assumes a structure like base_dir/cleaned_parsed_data/corpus_prefix/...
    """
    corpus_prefix_match = re.match(r'^([a-zA-Z0-9\-_]+?)_', doc_id) # Allow hyphen/underscore in prefix
    corpus_prefix = corpus_prefix_match.group(1) if corpus_prefix_match else ""
    source_sub_dir = "cleaned_parsed_data" # Configurable?

    possible_paths = []
    if corpus_prefix:
        cleaned_dir = os.path.join(base_dir, source_sub_dir, corpus_prefix)
        filename_part = doc_id.replace(f"{corpus_prefix}_", "", 1)
        possible_paths.extend([
            os.path.join(cleaned_dir, filename_part + ".txt"),
            os.path.join(cleaned_dir, doc_id + ".txt"),
        ])
    # Always check directly under source_sub_dir as fallback or if no prefix
    possible_paths.append(os.path.join(base_dir, source_sub_dir, doc_id + ".txt"))
    # Maybe even check base_dir directly?
    # possible_paths.append(os.path.join(base_dir, doc_id + ".txt"))


    for path in possible_paths:
        abs_path = os.path.abspath(path) # Check absolute path for clarity
        # print(f"DEBUG: Checking for original text at: {abs_path}") # Uncomment for deep debug
        if os.path.exists(abs_path):
            # print(f"DEBUG: Found original text at: {abs_path}") # Uncomment for deep debug
            return abs_path

    return None

# --- Main Processing Logic ---

def process_document(
    old_id: str, new_id: str, base_dir: str, output_base_dir: str,
    main_index: Dict[str, str], ner_index: Dict[str, str],
    main_env: str, ner_env: str, logger: Optional[FileOperationLogger] = None
) -> bool:
    """Process a single document."""
    overall_success = True
    temp_files_to_clean = []

    corpus_prefix_match = re.match(r'^([a-zA-Z0-9\-_]+?)_', old_id)
    corpus_prefix = corpus_prefix_match.group(1) if corpus_prefix_match else "unknown_corpus"

    # Base context for logging operations for this document
    log_context_base = {'old_id': old_id, 'new_id': new_id, 'corpus_prefix': corpus_prefix}

    if logger: logger.log_operation(**log_context_base, source_file="", destination_file="", operation_type="process_start", file_type="document", status="info", details=f"Started processing document {old_id} -> {new_id}")

    # --- 1. Locate Source Files ---
    main_docbin_path = main_index.get(old_id)
    ner_docbin_path = ner_index.get(old_id) # May be None

    if not main_docbin_path:
        if logger: logger.log_operation(**log_context_base, source_file="", destination_file="", operation_type="lookup", file_type="main_docbin", status="failed", details="Not found in main index.")
        return False
    if not os.path.exists(main_docbin_path):
        if logger: logger.log_operation(**log_context_base, source_file=main_docbin_path, destination_file="", operation_type="lookup", file_type="main_docbin", status="failed", details="Path not found on disk.")
        return False

    ner_docbin_path_resolved = None
    if ner_docbin_path:
        if os.path.exists(ner_docbin_path):
            ner_docbin_path_resolved = ner_docbin_path
        else:
             if logger: logger.log_operation(**log_context_base, source_file=ner_docbin_path, destination_file="", operation_type="lookup", file_type="ner_docbin", status="warning", details="NER index path not found on disk.")
    else:
         if logger: logger.log_operation(**log_context_base, source_file="", destination_file="", operation_type="lookup", file_type="ner_docbin", status="info", details="Not found in NER index.")


    source_txt_path = find_original_text(old_id, base_dir)
    if not source_txt_path and logger:
        logger.log_operation(**log_context_base, source_file=f"Searched in {base_dir}", destination_file="", operation_type="lookup", file_type="source_txt", status="warning", details="Original text file not found.")


    # --- 2. Create Output Directories ---
    output_dir = os.path.join(output_base_dir, corpus_prefix, new_id)
    texts_dir = os.path.join(output_dir, "texts")
    annotations_dir = os.path.join(output_dir, "annotations")
    try:
        os.makedirs(texts_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)
        # No need to log success here unless verbose needed
    except OSError as e:
         if logger: logger.log_operation(**log_context_base, source_file="", destination_file=output_dir, operation_type="create_dir", file_type="directory", status="failed", details=f"Error: {e}")
         return False

    # --- 3. Define Output File Paths ---
    output_txt = os.path.join(texts_dir, f"{new_id}.txt")
    output_txt_fullstop = os.path.join(texts_dir, f"{new_id}-fullstop.txt")
    output_csv_lemma = os.path.join(annotations_dir, f"{new_id}-lemma.csv")
    output_csv_upos = os.path.join(annotations_dir, f"{new_id}-upos.csv")
    output_csv_stop = os.path.join(annotations_dir, f"{new_id}-stop.csv")
    output_csv_dot = os.path.join(annotations_dir, f"{new_id}-dot.csv")
    output_csv_ner = os.path.join(annotations_dir, f"{new_id}-ner.csv")
    output_conllu = os.path.join(annotations_dir, f"{new_id}-conllu.conllu")
    temp_ner_tags_file = os.path.join(annotations_dir, f".{new_id}_temp_ner_tags.txt") # Hidden temp file
    temp_files_to_clean.append(temp_ner_tags_file)

    # --- 4. Perform Extractions and Operations ---
    try:
        # --- Main Env Operations ---
        if not extract_text_from_docbin(main_env, main_docbin_path, output_txt, logger, **log_context_base):
            overall_success = False
            # Don't attempt fullstop if text failed
            if logger: logger.log_operation(**log_context_base, source_file=output_txt, destination_file=output_txt_fullstop, operation_type="create", file_type="txt-fullstop", status="skipped", details="Input text failed extraction.")
        elif not create_fullstop_file(output_txt, output_txt_fullstop, logger, **log_context_base):
             pass # Logged inside function, maybe not critical failure

        if not extract_lemma_csv(main_env, main_docbin_path, output_csv_lemma, logger, **log_context_base): overall_success = False
        if not extract_upos_csv(main_env, main_docbin_path, output_csv_upos, logger, **log_context_base): overall_success = False
        if not extract_stop_csv(main_env, main_docbin_path, output_csv_stop, logger, **log_context_base): overall_success = False
        if not extract_dot_csv(main_env, main_docbin_path, output_csv_dot, logger, **log_context_base): overall_success = False

        # --- NER Env Operations ---
        ner_tags_available_for_conllu = False
        if ner_docbin_path_resolved:
            if not extract_ner_csv(ner_env, ner_docbin_path_resolved, output_csv_ner, logger, **log_context_base):
                pass # Logged inside, maybe not critical failure
            if extract_ner_tags_to_file(ner_env, ner_docbin_path_resolved, temp_ner_tags_file, logger, **log_context_base):
                ner_tags_available_for_conllu = True
            else:
                overall_success = False # Failed to get tags needed for CoNLL-U
        else:
             if logger: logger.log_operation(**log_context_base, source_file="", destination_file=output_csv_ner, operation_type="extract", file_type="csv-ner", status="skipped", details="NER docbin not found/resolved")

        # --- CoNLL-U Generation (Main Env + NER tags) ---
        ner_tags_input_path = temp_ner_tags_file if ner_tags_available_for_conllu else None
        if not extract_conllu_file(main_env, main_docbin_path, output_conllu, new_id, ner_tags_input_path, logger, **log_context_base):
             overall_success = False

        # --- Copy Original Text ---
        if source_txt_path:
            dest_original_txt = os.path.join(texts_dir, f"{new_id}-original.txt")
            try:
                shutil.copy2(source_txt_path, dest_original_txt)
                if logger: logger.log_operation(**log_context_base, source_file=source_txt_path, destination_file=dest_original_txt, operation_type="copy", file_type="source_txt", status="success")
            except Exception as e:
                if logger: logger.log_operation(**log_context_base, source_file=source_txt_path, destination_file=dest_original_txt, operation_type="copy", file_type="source_txt", status="failed", details=str(e))
                # overall_success = False # Decide if this failure is critical

    finally:
        # --- 5. Cleanup Temporary Files ---
        for temp_file in temp_files_to_clean:
            if os.path.exists(temp_file):
                try: os.remove(temp_file)
                except OSError as e:
                    if logger: logger.log_operation(**log_context_base, source_file=temp_file, destination_file="", operation_type="cleanup", file_type="temp_file", status="failed", details=f"Error: {e}")

    if logger: logger.log_operation(**log_context_base, source_file="", destination_file=output_dir, operation_type="process_end", file_type="document", status="success" if overall_success else "failed", details=f"Finished processing document {old_id} -> {new_id}. Overall success: {overall_success}")
    return overall_success


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize corpus documents based on CSV mapping and indices, extracting various formats including CoNLL-U with NER tags.")
    parser.add_argument("--mapping-csv", required=True, help="Path to the COMMA-delimited CSV file mapping old 'document_id' to new numeric 'sort_id'.")
    parser.add_argument("--main-index-csv", required=True, help="Path to the COMMA-delimited CSV index mapping 'document_id' to main DocBin file paths ('processed_path').") # Updated help text
    parser.add_argument("--ner-index-csv", required=True, help="Path to the COMMA-delimited CSV index mapping 'document_id' to NER DocBin file paths ('processed_path').") # Updated help text
    parser.add_argument("--base-dir", required=True, help="Path to the base directory containing original source data (e.g., 'cleaned_parsed_data' subdir).")
    parser.add_argument("--output-dir", required=True, help="Path to the base directory where reorganized output will be saved.")
    parser.add_argument("--main-env", required=True, help="Name of the Conda environment containing the main linguistic model (e.g., grc_proiel_trf).")
    parser.add_argument("--ner-env", required=True, help="Name of the Conda environment containing the NER model (e.g., grc_ner_trf).")
    parser.add_argument("--log-file", default="reorganization_log.csv", help="Path for the COMMA-delimited CSV log file (Default: ./reorganization_log.csv).")
    parser.add_argument("--wandb-project", default="corpus-reorganization", help="Name for the Weights & Biases project.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")

    args = parser.parse_args()

    start_time = time.time()

    # Initialize Logger (Handles internal errors gracefully)
    logger = FileOperationLogger(
        log_file_path=args.log_file,
        use_wandb=(not args.no_wandb),
        wandb_project=args.wandb_project
    )

    print(f"Starting corpus reorganization process.")
    # Print config for clarity
    print("-" * 30)
    for arg, value in vars(args).items():
        print(f"{arg:<20}: {value}")
    print("-" * 30)

    if logger.use_wandb:
         try:
              wandb.config.update(vars(args)) # Log command line args to wandb config
         except Exception as e:
              print(f"!!! Warning: Failed to update wandb config: {e}")

    # Load Mappings and Indices
    print("Loading mappings and indices...")
    mappings = parse_csv_mapping(args.mapping_csv)
    main_index = load_index(args.main_index_csv)
    ner_index = load_index(args.ner_index_csv)
    print(f"Loaded {len(mappings)} ID mappings.")
    print(f"Loaded {len(main_index)} main index entries.")
    print(f"Loaded {len(ner_index)} NER index entries.")

    if not mappings or not main_index:
        print("\nError: Mapping CSV or Main Index CSV could not be loaded or are empty. Exiting.")
        if logger.use_wandb:
             try: wandb.finish(exit_code=1)
             except Exception: pass
        exit(1)

    # Process Documents
    processed_count = 0
    failed_count = 0
    print(f"\nProcessing {len(mappings)} documents...")

    for old_id, new_id in tqdm.tqdm(mappings.items(), desc="Processing Documents", unit="doc"):
        # Check if main docbin path actually exists before processing
        main_docbin_path_check = main_index.get(old_id)
        if not main_docbin_path_check or not os.path.exists(main_docbin_path_check):
             print(f"\nSkipping {old_id} -> {new_id}: Main DocBin path missing or invalid in index.")
             if logger:
                 logger.log_operation(old_id=old_id, new_id=new_id, corpus_prefix=(old_id.split('_')[0] if '_' in old_id else 'unknown'),
                                      source_file=main_docbin_path_check or "", destination_file="",
                                      operation_type="process_skip", file_type="document", status="warning",
                                      details="Main DocBin path missing or invalid in index.")
             failed_count += 1
             processed_count += 1 # Count as attempted
             continue # Skip to next document

        # Proceed with processing
        success = process_document(
            old_id=old_id, new_id=new_id, base_dir=args.base_dir, output_base_dir=args.output_dir,
            main_index=main_index, ner_index=ner_index, main_env=args.main_env, ner_env=args.ner_env, logger=logger
        )
        if not success:
            failed_count += 1
        processed_count += 1

    # Completion Summary
    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "-"*30)
    print("--- Reorganization Summary ---")
    print(f"Total documents attempted: {processed_count}")
    print(f"Successfully processed: {processed_count - failed_count}")
    print(f"Failed documents: {failed_count}")
    print(f"Total time taken: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print("-" * 30)

    # Summarize and close logger (logs final summary to wandb)
    summary_stats = logger.summarize_and_close()
    if logger.log_file_path: print(f"Detailed log saved to: {logger.log_file_path}")
    if logger.use_wandb:
        run_url = wandb.run.get_url() if wandb.run else None # Get URL before finish potentially closes run
        if run_url: print(f"W&B Run URL: {run_url}")
        else: print("W&B logging was enabled but run URL not available.")

    print("\nScript finished.")
