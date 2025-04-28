#!/usr/bin/env python3
"""
Script to reorganize corpus documents according to the database schema.
Reads document mappings from CSV and uses indices to locate docbin files.
Extracts various formats (TXT, CSVs, CoNLL-U) using specified spaCy models
in separate conda environments. Includes NER tags in CoNLL-U MISC column.
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
import tempfile # Added for temporary file handling
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import tqdm # Keep tqdm for progress bars
import wandb


class FileOperationLogger:
    """
    Logger to track file operations during the document extraction process.
    """
    def __init__(self, log_file_path: str, use_wandb: bool = True, wandb_project: str = "corpus-reorganization"):
        """
        Initialize the logger.

        Args:
            log_file_path: Path to the CSV log file (using comma delimiter)
            use_wandb: Whether to use wandb for logging
            wandb_project: Name of the wandb project
        """
        self.log_file_path = log_file_path
        self.log_entries = []
        self.use_wandb = use_wandb

        # Create log file with headers (using comma delimiter)
        with open(log_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Using csv.writer defaults to comma delimiter
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp', 'old_id', 'new_id', 'corpus_prefix',
                'source_file', 'destination_file', 'operation_type',
                'file_type', 'status', 'details'
            ])

        # Initialize wandb if enabled
        if use_wandb:
            wandb.init(project=wandb_project, name=f"reorganization-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            wandb.config.update({
                "log_file": log_file_path,
                "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

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
        """
        Log a file operation.

        Args:
            old_id: Original document identifier
            new_id: New numeric identifier
            corpus_prefix: Corpus prefix (e.g., 'perseus')
            source_file: Source file path
            destination_file: Destination file path
            operation_type: Type of operation (extract, copy, create, lookup)
            file_type: Type of file (txt, csv-lemma, conllu, ner-tags, etc.)
            status: Operation status (success, failed, warning, info)
            details: Additional details or error message
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        entry = {
            'timestamp': timestamp,
            'old_id': old_id,
            'new_id': new_id,
            'corpus_prefix': corpus_prefix,
            'source_file': str(source_file), # Ensure paths are strings
            'destination_file': str(destination_file), # Ensure paths are strings
            'operation_type': operation_type,
            'file_type': file_type,
            'status': status,
            'details': details
        }

        self.log_entries.append(entry)

        # Write to CSV immediately (using comma delimiter)
        with open(self.log_file_path, 'a', newline='', encoding='utf-8') as csvfile:
             # Using csv.writer defaults to comma delimiter
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp, old_id, new_id, corpus_prefix,
                entry['source_file'], entry['destination_file'], operation_type,
                file_type, status, details
            ])

        # Log to wandb if enabled
        if self.use_wandb:
            # Log individual operations
            wandb.log({
                "file_operation": entry,
                "document_processed": f"{old_id} -> {new_id}",
                "operation_status": status == "success"
            })

    def summarize_and_close(self) -> Dict[str, Any]:
        """
        Summarize the operations and close the logger.

        Returns:
            Dictionary with summary statistics
        """
        # Calculate summary statistics
        total_operations = len(self.log_entries)
        successful_operations = sum(1 for entry in self.log_entries if entry['status'] == 'success')
        failed_operations = sum(1 for entry in self.log_entries if entry['status'] == 'failed')
        warning_operations = sum(1 for entry in self.log_entries if entry['status'] == 'warning')


        # Group by operation type
        operation_counts = {}
        for entry in self.log_entries:
            op_type = entry['operation_type']
            operation_counts[op_type] = operation_counts.get(op_type, 0) + 1

        # Group by file type
        file_type_counts = {}
        for entry in self.log_entries:
            file_type = entry['file_type']
            file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1

        # Count unique documents processed (based on start log entry)
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
            'unique_new_ids_created': len(unique_new_ids)
        }

        # Log summary to wandb
        if self.use_wandb:
            wandb.log(summary)
            wandb.log({
                "success_rate": successful_operations / total_operations if total_operations > 0 else 0,
                "completion_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            # Create and log summary table
            columns = ["Metric", "Value"]
            data = [
                ["Total Operations", total_operations],
                ["Successful Operations", successful_operations],
                ["Failed Operations", failed_operations],
                ["Warning Operations", warning_operations],
                ["Unique Documents Processed", len(unique_old_ids)],
                ["Unique New IDs Created", len(unique_new_ids)]
            ]

            # Add operation type counts
            for op_type, count in operation_counts.items():
                data.append([f"Operation: {op_type}", count])

            # Add file type counts
            for file_type, count in file_type_counts.items():
                data.append([f"File Type: {file_type}", count])

            summary_table = wandb.Table(columns=columns, data=data)
            wandb.log({"summary_statistics": summary_table})

            wandb.finish()

        return summary


def extract_text_from_docbin(
    conda_env: str, # Environment with grc_proiel_trf
    docbin_path: str,
    output_txt_path: str,
    logger: Optional[FileOperationLogger] = None,
    old_id: str = "",
    new_id: str = "",
    corpus_prefix: str = ""
) -> bool:
    """
    Extract plain text from DocBin using the main linguistic model.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Uses grc_proiel_trf by default, as it's the main environment's model
        cmd = f'conda run -n {conda_env} python -c "import spacy; from spacy.tokens import DocBin; nlp = spacy.load(\'grc_proiel_trf\'); doc_bin = DocBin().from_disk(\'{docbin_path}\'); doc = list(doc_bin.get_docs(nlp.vocab))[0]; with open(\'{output_txt_path}\', \'w\', encoding=\'utf-8\') as f: f.write(doc.text)"'
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True) # Added capture_output

        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_txt_path,
                operation_type="extract", file_type="txt", status="success"
            )
        return True
    except subprocess.CalledProcessError as e:
        error_details = f"Command failed: {e.cmd}\nStderr: {e.stderr}\nStdout: {e.stdout}"
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_txt_path,
                operation_type="extract", file_type="txt", status="failed",
                details=error_details
            )
        return False
    except Exception as e:
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_txt_path,
                operation_type="extract", file_type="txt", status="failed",
                details=f"Unexpected error: {str(e)}"
            )
        return False


def create_fullstop_file(
    input_txt_path: str,
    output_txt_path: str,
    logger: Optional[FileOperationLogger] = None,
    old_id: str = "",
    new_id: str = "",
    corpus_prefix: str = ""
) -> bool:
    """
    Create a fullstop file where each sentence (ending with '.') is on a new line.

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(input_txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Replace periods that are likely sentence endings with period + newline
        # Avoids matching ellipses (...) or decimals (though less likely in corpus).
        # Basic approach: replace '.' not followed by another '.'
        text = re.sub(r'\.(?!\.)', '.\n', text)
        # Optional: Handle other punctuation like '?' and '!' if needed
        # text = re.sub(r'[.?!](?![.?!])', lambda m: m.group(0) + '\n', text)

        # Clean up extra whitespace around newlines
        text = re.sub(r'\s+\n', '\n', text)
        text = re.sub(r'\n\s+', '\n', text).strip()


        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=input_txt_path, destination_file=output_txt_path,
                operation_type="create", file_type="txt-fullstop", status="success"
            )
        return True
    except Exception as e:
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=input_txt_path, destination_file=output_txt_path,
                operation_type="create", file_type="txt-fullstop", status="failed",
                details=str(e)
            )
        return False


def extract_lemma_csv(
    conda_env: str, # Environment with grc_proiel_trf
    docbin_path: str,
    output_csv_path: str,
    logger: Optional[FileOperationLogger] = None,
    old_id: str = "",
    new_id: str = "",
    corpus_prefix: str = ""
) -> bool:
    """
    Extract lemma information to CSV (comma-delimited) from DocBin using main model.
    Includes 1-based token ID.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Uses grc_proiel_trf. Creates comma-delimited CSV.
        cmd = f'conda run -n {conda_env} python -c "import csv; import spacy; from spacy.tokens import DocBin; nlp = spacy.load(\'grc_proiel_trf\'); doc_bin = DocBin().from_disk(\'{docbin_path}\'); doc = list(doc_bin.get_docs(nlp.vocab))[0]; with open(\'{output_csv_path}\', \'w\', encoding=\'utf-8\', newline=\'\') as csvfile: writer = csv.writer(csvfile); writer.writerow([\'ID\', \'TOKEN\', \'LEMMA\']); count = 0; for token in doc: count += 1; writer.writerow([count, token.text, token.lemma_])"'
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_csv_path,
                operation_type="extract", file_type="csv-lemma", status="success"
            )
        return True
    except subprocess.CalledProcessError as e:
        error_details = f"Command failed: {e.cmd}\nStderr: {e.stderr}\nStdout: {e.stdout}"
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_csv_path,
                operation_type="extract", file_type="csv-lemma", status="failed",
                details=error_details
            )
        return False
    except Exception as e:
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_csv_path,
                operation_type="extract", file_type="csv-lemma", status="failed",
                details=f"Unexpected error: {str(e)}"
            )
        return False


def extract_upos_csv(
    conda_env: str, # Environment with grc_proiel_trf
    docbin_path: str,
    output_csv_path: str,
    logger: Optional[FileOperationLogger] = None,
    old_id: str = "",
    new_id: str = "",
    corpus_prefix: str = ""
) -> bool:
    """
    Extract UPOS information to CSV (comma-delimited) from DocBin using main model.
    Includes 1-based token ID.

    Returns:
        True if successful, False otherwise
    """
    try:
         # Uses grc_proiel_trf. Creates comma-delimited CSV.
        cmd = f'conda run -n {conda_env} python -c "import csv; import spacy; from spacy.tokens import DocBin; nlp = spacy.load(\'grc_proiel_trf\'); doc_bin = DocBin().from_disk(\'{docbin_path}\'); doc = list(doc_bin.get_docs(nlp.vocab))[0]; with open(\'{output_csv_path}\', \'w\', encoding=\'utf-8\', newline=\'\') as csvfile: writer = csv.writer(csvfile); writer.writerow([\'ID\', \'TOKEN\', \'UPOS\']); count = 0; for token in doc: count += 1; writer.writerow([count, token.text, token.pos_])"'
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_csv_path,
                operation_type="extract", file_type="csv-upos", status="success"
            )
        return True
    except subprocess.CalledProcessError as e:
        error_details = f"Command failed: {e.cmd}\nStderr: {e.stderr}\nStdout: {e.stdout}"
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_csv_path,
                operation_type="extract", file_type="csv-upos", status="failed",
                details=error_details
            )
        return False
    except Exception as e:
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_csv_path,
                operation_type="extract", file_type="csv-upos", status="failed",
                details=f"Unexpected error: {str(e)}"
            )
        return False


def extract_stop_csv(
    conda_env: str, # Environment with grc_proiel_trf
    docbin_path: str,
    output_csv_path: str,
    logger: Optional[FileOperationLogger] = None,
    old_id: str = "",
    new_id: str = "",
    corpus_prefix: str = ""
) -> bool:
    """
    Extract stopword information to CSV (comma-delimited) from DocBin using main model.
    Includes 1-based token ID.

    Returns:
        True if successful, False otherwise
    """
    try:
         # Uses grc_proiel_trf. Creates comma-delimited CSV.
        cmd = f'conda run -n {conda_env} python -c "import csv; import spacy; from spacy.tokens import DocBin; nlp = spacy.load(\'grc_proiel_trf\'); doc_bin = DocBin().from_disk(\'{docbin_path}\'); doc = list(doc_bin.get_docs(nlp.vocab))[0]; with open(\'{output_csv_path}\', \'w\', encoding=\'utf-8\', newline=\'\') as csvfile: writer = csv.writer(csvfile); writer.writerow([\'ID\', \'TOKEN\', \'IS_STOP\']); count = 0; for token in doc: count += 1; writer.writerow([count, token.text, \'TRUE\' if token.is_stop else \'FALSE\'])"'
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_csv_path,
                operation_type="extract", file_type="csv-stop", status="success"
            )
        return True
    except subprocess.CalledProcessError as e:
        error_details = f"Command failed: {e.cmd}\nStderr: {e.stderr}\nStdout: {e.stdout}"
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_csv_path,
                operation_type="extract", file_type="csv-stop", status="failed",
                details=error_details
            )
        return False
    except Exception as e:
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_csv_path,
                operation_type="extract", file_type="csv-stop", status="failed",
                details=f"Unexpected error: {str(e)}"
            )
        return False


# --- NER Extraction (uses ner_env) ---

def extract_ner_csv(
    conda_env: str, # Environment with grc_ner_trf
    docbin_path: str,
    output_csv_path: str,
    logger: Optional[FileOperationLogger] = None,
    old_id: str = "",
    new_id: str = "",
    corpus_prefix: str = ""
) -> bool:
    """
    Extract NER information to CSV (comma-delimited) from DocBin using NER model.
    Includes 1-based token ID.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Uses grc_ner_trf. Creates comma-delimited CSV.
        cmd = f'conda run -n {conda_env} python -c "import csv; import spacy; from spacy.tokens import DocBin; nlp = spacy.load(\'grc_ner_trf\'); doc_bin = DocBin().from_disk(\'{docbin_path}\'); doc = list(doc_bin.get_docs(nlp.vocab))[0]; with open(\'{output_csv_path}\', \'w\', encoding=\'utf-8\', newline=\'\') as csvfile: writer = csv.writer(csvfile); writer.writerow([\'ID\', \'TOKEN\', \'NER\']); count = 0; for token in doc: count += 1; writer.writerow([count, token.text, token.ent_type_ if token.ent_type_ else \'O\'])"'
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_csv_path,
                operation_type="extract", file_type="csv-ner", status="success"
            )
        return True
    except subprocess.CalledProcessError as e:
        error_details = f"Command failed: {e.cmd}\nStderr: {e.stderr}\nStdout: {e.stdout}"
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_csv_path,
                operation_type="extract", file_type="csv-ner", status="failed",
                details=error_details
            )
        return False
    except Exception as e:
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_csv_path,
                operation_type="extract", file_type="csv-ner", status="failed",
                details=f"Unexpected error: {str(e)}"
            )
        return False

# --- New function to extract ONLY NER tags to a file ---
def extract_ner_tags_to_file(
    conda_env: str, # Environment with grc_ner_trf
    ner_docbin_path: str,
    output_tags_path: str,
    logger: Optional[FileOperationLogger] = None,
    old_id: str = "",
    new_id: str = "",
    corpus_prefix: str = ""
) -> bool:
    """
    Extract only NER tags (one per line) to a file using the NER model.

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Uses grc_ner_trf
        cmd = f'conda run -n {conda_env} python -c "import spacy; from spacy.tokens import DocBin; nlp = spacy.load(\'grc_ner_trf\'); doc_bin = DocBin().from_disk(\'{ner_docbin_path}\'); doc = list(doc_bin.get_docs(nlp.vocab))[0]; with open(\'{output_tags_path}\', \'w\', encoding=\'utf-8\') as f: f.write(\'\\n\'.join([token.ent_type_ if token.ent_type_ else \'O\' for token in doc]))"'
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=ner_docbin_path, destination_file=output_tags_path,
                operation_type="extract", file_type="ner-tags", status="success"
            )
        return True
    except subprocess.CalledProcessError as e:
        error_details = f"Command failed: {e.cmd}\nStderr: {e.stderr}\nStdout: {e.stdout}"
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=ner_docbin_path, destination_file=output_tags_path,
                operation_type="extract", file_type="ner-tags", status="failed",
                details=error_details
            )
        return False
    except Exception as e:
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=ner_docbin_path, destination_file=output_tags_path,
                operation_type="extract", file_type="ner-tags", status="failed",
                details=f"Unexpected error: {str(e)}"
            )
        return False


# --- Punctuation Extraction (uses main_env) ---

def extract_dot_csv(
    conda_env: str, # Environment with grc_proiel_trf
    docbin_path: str,
    output_csv_path: str,
    logger: Optional[FileOperationLogger] = None,
    old_id: str = "",
    new_id: str = "",
    corpus_prefix: str = ""
) -> bool:
    """
    Extract punctuation information to CSV (comma-delimited) from DocBin using main model.
    Includes 1-based token ID.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Uses grc_proiel_trf. Creates comma-delimited CSV.
        cmd = f'conda run -n {conda_env} python -c "import csv; import spacy; from spacy.tokens import DocBin; nlp = spacy.load(\'grc_proiel_trf\'); doc_bin = DocBin().from_disk(\'{docbin_path}\'); doc = list(doc_bin.get_docs(nlp.vocab))[0]; with open(\'{output_csv_path}\', \'w\', encoding=\'utf-8\', newline=\'\') as csvfile: writer = csv.writer(csvfile); writer.writerow([\'ID\', \'TOKEN\', \'IS_PUNCT\']); count = 0; for token in doc: count += 1; writer.writerow([count, token.text, \'TRUE\' if token.is_punct else \'FALSE\'])"'
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_csv_path,
                operation_type="extract", file_type="csv-dot", status="success"
            )
        return True
    except subprocess.CalledProcessError as e:
        error_details = f"Command failed: {e.cmd}\nStderr: {e.stderr}\nStdout: {e.stdout}"
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_csv_path,
                operation_type="extract", file_type="csv-dot", status="failed",
                details=error_details
            )
        return False
    except Exception as e:
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=docbin_path, destination_file=output_csv_path,
                operation_type="extract", file_type="csv-dot", status="failed",
                details=f"Unexpected error: {str(e)}"
            )
        return False


# --- CoNLL-U Extraction (uses main_env, incorporates NER tags) ---

def extract_conllu_file(
    conda_env: str, # Environment with grc_proiel_trf
    main_docbin_path: str,
    output_conllu_path: str,
    doc_id: str, # The NEW numeric ID
    ner_tags_path: Optional[str], # Path to the file with NER tags (or None)
    logger: Optional[FileOperationLogger] = None,
    old_id: str = "",
    new_id: str = "", # Redundant with doc_id but kept for consistency
    corpus_prefix: str = ""
) -> bool:
    """
    Extract CoNLL-U format from DocBin using the main model.
    If ner_tags_path is provided, reads NER tags from it and adds them
    to the MISC column (as NER=TAG).

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create a temporary script to extract conllu, now incorporating NER tags
        script_content = f"""
import sys
import spacy
import os # Added
from spacy.tokens import DocBin

# --- Configuration ---
main_docbin_path = r'{main_docbin_path}' # Use raw string for paths
output_conllu_path = r'{output_conllu_path}'
ner_tags_path = {repr(ner_tags_path)} # Use repr to handle None or path string correctly
doc_id_str = '{doc_id}'
# --- End Configuration ---


# Load main linguistic model (proiel)
nlp = spacy.load('grc_proiel_trf')

# Load DocBin from main model
doc_bin = DocBin().from_disk(main_docbin_path)
doc = list(doc_bin.get_docs(nlp.vocab))[0]

# Load NER tags if path is provided
ner_tags = None
if ner_tags_path and os.path.exists(ner_tags_path):
    try:
        with open(ner_tags_path, 'r', encoding='utf-8') as f_ner:
            ner_tags = [line.strip() for line in f_ner if line.strip()]
        # Basic validation: check if number of tags matches number of tokens
        if len(ner_tags) != len(doc):
            print(f"Warning: Mismatch between token count ({{len(doc)}}) and NER tag count ({{len(ner_tags)}}) for {{doc_id_str}}. NER tags in CoNLL-U might be incorrect.", file=sys.stderr)
            # Decide how to handle: proceed with potential error, or disable NER?
            # For now, let's proceed but the warning is printed.
            # ner_tags = None # Uncomment this line to disable NER tags on mismatch
    except Exception as e:
        print(f"Warning: Could not read NER tags from {{ner_tags_path}}: {{e}}. NER tags will not be included.", file=sys.stderr)
        ner_tags = None
elif ner_tags_path:
     print(f"Warning: Provided NER tags path does not exist: {{ner_tags_path}}. NER tags will not be included.", file=sys.stderr)


# Extract CoNLL-U format
with open(output_conllu_path, "w", encoding="utf-8") as f_out:
    # Write document header
    f_out.write(f"# newdoc id = {{doc_id_str}}\\n")

    sent_id_counter = 1
    token_doc_offset = 0 # Keep track of token index within the doc

    for sent in doc.sents:
        # Write sentence header
        # Use consistent sentence ID format if possible, e.g., doc_id-s1, doc_id-s2
        f_out.write(f"# sent_id = {{doc_id_str}}-{{sent_id_counter}}\\n")
        f_out.write(f"# text = {{sent.text.replace('\\n', ' ')}}\\n") # Replace potential newlines in sentence text

        token_sent_id_counter = 1
        for token in sent:
            # Format: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
            # ID: Token ID relative to the sentence (1-based)
            # HEAD: Head token ID relative to the sentence (0 for root)
            head_id = token.head.i - sent.start + 1 if token.head.i != token.i else 0

            # Handle morphological features
            feats = str(token.morph) if token.morph else "_"
            if feats == "": feats = "_" # Ensure it's "_" if empty string

            # --- MISC Column Handling ---
            misc_parts = []
            # Add NER tag if available
            if ner_tags:
                try:
                    # token.i is the index within the *document*
                    ner_tag = ner_tags[token.i]
                    if ner_tag != 'O': # Only add if not 'O' for brevity? Optional choice.
                         misc_parts.append(f"NER={{ner_tag}}")
                    # If you want to always include NER=O:
                    # misc_parts.append(f"NER={{ner_tag}}")
                except IndexError:
                     # This happens if counts mismatched and we decided to proceed
                     misc_parts.append("NER=Error")


            # Add SpaceAfter=No if applicable (check if next token exists and has no separating whitespace)
            if token.i + 1 < len(doc) and doc[token.i+1].idx == token.idx + len(token.text):
                 misc_parts.append("SpaceAfter=No")

            # Combine MISC parts or use "_"
            misc_field = "|".join(misc_parts) if misc_parts else "_"
            # --- End MISC Column Handling ---


            # DEPS and MISC are often simplified or generated by more complex tools
            # Using "_" as placeholders here.
            # Note: CoNLL-U columns are TAB-separated
            line = (f"{{token_sent_id_counter}}\\t{{token.text}}\\t{{token.lemma_}}\\t{{token.pos_}}\\t"
                   f"{{token.tag_}}\\t{{feats}}\\t{{head_id}}\\t"
                   f"{{token.dep_}}\\t_\\t{{misc_field}}\\n") # Added misc_field
            f_out.write(line)
            token_sent_id_counter += 1
            token_doc_offset += 1

        f_out.write("\\n")  # Empty line between sentences
        sent_id_counter += 1

"""
        # Use tempfile for the script itself
        temp_script_fd, temp_script_path = tempfile.mkstemp(suffix='.py', text=True)
        os.close(temp_script_fd) # Close file descriptor early

        try:
            with open(temp_script_path, 'w', encoding='utf-8') as f_script:
                f_script.write(script_content)

            # Run script in conda environment
            cmd = f"conda run -n {conda_env} python {temp_script_path}"
            # Increased timeout potential needed for larger files/models
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=600) # Added capture_output, text, timeout

            # Log warnings from the script's stderr if any
            if result.stderr and logger:
                 logger.log_operation(
                    old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                    source_file=main_docbin_path, destination_file=output_conllu_path,
                    operation_type="extract", file_type="conllu", status="warning",
                    details=f"CoNLL-U script stderr: {result.stderr.strip()}"
                )

            if logger:
                logger.log_operation(
                    old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                    source_file=main_docbin_path, destination_file=output_conllu_path,
                    operation_type="extract", file_type="conllu", status="success",
                    details=f"NER tags from '{ner_tags_path}' included." if ner_tags_path else "No NER tags included."
                )
            return True
        except subprocess.TimeoutExpired as e:
             error_details = f"Command timed out ({e.timeout}s): {e.cmd}\nStderr: {e.stderr}\nStdout: {e.stdout}"
             if logger:
                 logger.log_operation(
                     old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                     source_file=main_docbin_path, destination_file=output_conllu_path,
                     operation_type="extract", file_type="conllu", status="failed",
                     details=error_details
                 )
             return False
        except subprocess.CalledProcessError as e:
            error_details = f"Command failed: {e.cmd}\nStderr: {e.stderr}\nStdout: {e.stdout}"
            if logger:
                logger.log_operation(
                    old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                    source_file=main_docbin_path, destination_file=output_conllu_path,
                    operation_type="extract", file_type="conllu", status="failed",
                    details=error_details
                )
            return False
        except Exception as e:
            if logger:
                logger.log_operation(
                    old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                    source_file=main_docbin_path, destination_file=output_conllu_path,
                    operation_type="extract", file_type="conllu", status="failed",
                    details=f"Unexpected error: {str(e)}"
                )
            return False
        finally:
            # Clean up temp script
            if os.path.exists(temp_script_path):
                os.unlink(temp_script_path)

    except Exception as e: # Catch errors in setting up the temp script itself
         if logger:
             logger.log_operation(
                 old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                 source_file=main_docbin_path, destination_file=output_conllu_path,
                 operation_type="extract", file_type="conllu", status="failed",
                 details=f"Error setting up CoNLL-U extraction script: {str(e)}"
             )
         return False


# --- Helper Functions ---

def load_index(index_path: str) -> Dict[str, str]:
    """
    Load an index file (CSV, comma-delimited) that maps document IDs to file paths.
    Expects columns 'document_id' and 'processed_path'.
    """
    index = {}
    try:
        # pandas defaults to comma delimiter
        df = pd.read_csv(index_path)

        # --- MODIFIED LINES START ---
        # Check for 'document_id' and 'processed_path' columns
        required_path_col = 'processed_path' # Use the correct column name
        if 'document_id' in df.columns and required_path_col in df.columns:
            for _, row in df.iterrows():
                # Convert IDs to string and strip whitespace just in case
                doc_id = str(row['document_id']).strip()
                # Access the path using the correct column name
                file_path = str(row[required_path_col]).strip()
                index[doc_id] = file_path
        else:
            # Updated warning message
            print(f"Warning: Index file '{index_path}' missing required columns 'document_id' or '{required_path_col}'.")
        # --- MODIFIED LINES END ---

    except Exception as e:
         print(f"Error loading index file '{index_path}': {e}")
         # Depending on severity, you might want to raise the exception
         # raise
    return index


def parse_csv_mapping(csv_path: str) -> Dict[str, str]:
    """
    Parse the CSV file (comma-delimited) containing old to new ID mappings.
    Returns a dictionary mapping old IDs (str) to new numeric IDs (str).
    """
    mappings = {}
    try:
        # pandas defaults to comma delimiter
        df = pd.read_csv(csv_path)

        # Extract document_id and corresponding sort_id
        if 'document_id' in df.columns and 'sort_id' in df.columns:
             for index, row in df.iterrows():
                # Convert IDs to string and strip whitespace
                old_id = str(row['document_id']).strip()
                # Ensure sort_id is treated as integer then string, handle missing
                new_id = str(int(row['sort_id'])) if pd.notna(row['sort_id']) else None

                if old_id and new_id:
                    mappings[old_id] = new_id
                elif old_id and not new_id:
                    print(f"Warning: Missing sort_id for document_id '{old_id}' in '{csv_path}'. Skipping.")
        else:
             print(f"Warning: Mapping file '{csv_path}' missing required columns 'document_id' or 'sort_id'.")

    except Exception as e:
         print(f"Error parsing mapping file '{csv_path}': {e}")
         # raise
    return mappings


def find_original_text(doc_id: str, base_dir: str) -> Optional[str]:
    """
    Try to locate the original text file for a document ID.
    Assumes a structure like base_dir/cleaned_parsed_data/corpus_prefix/...
    """
    # Extract corpus prefix (e.g., 'perseus')
    corpus_prefix_match = re.match(r'^([a-zA-Z0-9]+)_', doc_id)
    corpus_prefix = corpus_prefix_match.group(1) if corpus_prefix_match else ""

    if not corpus_prefix:
        print(f"Warning: Could not determine corpus prefix from doc_id '{doc_id}' for finding original text.")
        # Try looking directly in cleaned_parsed_data without prefix?
        possible_paths = [
             os.path.join(base_dir, "cleaned_parsed_data", doc_id + ".txt")
        ]
    else:
         # Possible locations for the original text
        cleaned_dir = os.path.join(base_dir, "cleaned_parsed_data", corpus_prefix)
        # Try removing prefix from doc_id for filename, and also using full doc_id
        filename_part = doc_id.replace(f"{corpus_prefix}_", "", 1)
        possible_paths = [
            os.path.join(cleaned_dir, filename_part + ".txt"),
            os.path.join(cleaned_dir, doc_id + ".txt"),
        ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # Fallback: Maybe it's directly under cleaned_parsed_data?
    fallback_path = os.path.join(base_dir, "cleaned_parsed_data", doc_id + ".txt")
    if os.path.exists(fallback_path):
        return fallback_path

    return None


# --- Main Processing Logic ---

def process_document(
    old_id: str,
    new_id: str,
    base_dir: str,
    output_base_dir: str,
    main_index: Dict[str, str],
    ner_index: Dict[str, str],
    main_env: str, # e.g., 'proiel_trf'
    ner_env: str,  # e.g., 'ner'
    logger: Optional[FileOperationLogger] = None
) -> bool:
    """
    Process a single document: find files, extract formats, save to new structure.

    Returns:
        True if all essential operations were successful, False otherwise.
    """
    overall_success = True # Track if all steps succeed for this doc
    temp_files_to_clean = [] # Keep track of temporary files

    # Determine corpus prefix for logging and output path
    corpus_prefix_match = re.match(r'^([a-zA-Z0-9]+)_', old_id)
    corpus_prefix = corpus_prefix_match.group(1) if corpus_prefix_match else "unknown_corpus"
    if corpus_prefix == "unknown_corpus" and logger:
         logger.log_operation(old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix, source_file="", destination_file="", operation_type="parse_id", file_type="document", status="warning", details="Could not determine corpus prefix from old_id")


    if logger:
        logger.log_operation(
            old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
            source_file="", destination_file="", operation_type="process_start",
            file_type="document", status="info",
            details=f"Started processing document {old_id} -> {new_id}"
        )

    # --- 1. Locate Source Files ---
    main_docbin_path = main_index.get(old_id)
    ner_docbin_path = ner_index.get(old_id)

    if not main_docbin_path:
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file="", destination_file="", operation_type="lookup",
                file_type="main_docbin", status="failed",
                details=f"No main docbin found for {old_id} in main index. Cannot proceed."
            )
        return False # Cannot proceed without main docbin

    if not os.path.exists(main_docbin_path):
         if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=main_docbin_path, destination_file="", operation_type="lookup",
                file_type="main_docbin", status="failed",
                details=f"Main docbin path not found on disk: {main_docbin_path}. Cannot proceed."
            )
         return False

    if not ner_docbin_path:
        if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file="", destination_file="", operation_type="lookup",
                file_type="ner_docbin", status="warning",
                details=f"No NER docbin found for {old_id} in NER index. NER features will be omitted."
            )
    elif not os.path.exists(ner_docbin_path):
         if logger:
            logger.log_operation(
                old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                source_file=ner_docbin_path, destination_file="", operation_type="lookup",
                file_type="ner_docbin", status="warning",
                details=f"NER docbin path not found on disk: {ner_docbin_path}. NER features will be omitted."
            )
         ner_docbin_path = None # Treat as if not found


    # Find original text file if available
    source_txt_path = find_original_text(old_id, base_dir)
    if source_txt_path and logger:
        logger.log_operation(
            old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
            source_file=source_txt_path, destination_file="", operation_type="lookup",
            file_type="source_txt", status="success", details="Found original text file."
        )
    elif logger:
        logger.log_operation(
            old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
            source_file="", destination_file="", operation_type="lookup",
            file_type="source_txt", status="warning", details="No original text file found."
        )

    # --- 2. Create Output Directories ---
    output_dir = os.path.join(output_base_dir, corpus_prefix, new_id)
    texts_dir = os.path.join(output_dir, "texts")
    annotations_dir = os.path.join(output_dir, "annotations")
    try:
        os.makedirs(texts_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)
        if logger:
             logger.log_operation(
                 old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                 source_file="", destination_file=output_dir, operation_type="create_dir",
                 file_type="directory", status="success"
             )
    except OSError as e:
         if logger:
             logger.log_operation(
                 old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                 source_file="", destination_file=output_dir, operation_type="create_dir",
                 file_type="directory", status="failed", details=f"Error creating directory: {e}"
             )
         return False # Cannot proceed if directories cannot be created

    # --- 3. Define Output File Paths ---
    output_txt = os.path.join(texts_dir, f"{new_id}.txt")
    output_txt_fullstop = os.path.join(texts_dir, f"{new_id}-fullstop.txt")
    output_csv_lemma = os.path.join(annotations_dir, f"{new_id}-lemma.csv")
    output_csv_upos = os.path.join(annotations_dir, f"{new_id}-upos.csv")
    output_csv_stop = os.path.join(annotations_dir, f"{new_id}-stop.csv")
    output_csv_dot = os.path.join(annotations_dir, f"{new_id}-dot.csv")
    output_csv_ner = os.path.join(annotations_dir, f"{new_id}-ner.csv")
    output_conllu = os.path.join(annotations_dir, f"{new_id}-conllu.conllu")
    # Temporary file for NER tags (will be cleaned up)
    temp_ner_tags_file = os.path.join(annotations_dir, f"{new_id}_temp_ner_tags.txt")
    temp_files_to_clean.append(temp_ner_tags_file)


    # --- 4. Perform Extractions and Operations ---
    try:
        # Extract plain text (using main env)
        if not extract_text_from_docbin(main_env, main_docbin_path, output_txt, logger, old_id, new_id, corpus_prefix):
            overall_success = False

        # Create fullstop version (if text extraction succeeded)
        if os.path.exists(output_txt):
            if not create_fullstop_file(output_txt, output_txt_fullstop, logger, old_id, new_id, corpus_prefix):
                 overall_success = False # Mark as non-critical? Maybe just warning. Logged inside func.
        else:
            if logger: logger.log_operation(old_id,new_id,corpus_prefix,output_txt, output_txt_fullstop,"create","txt-fullstop","skipped","Input text file missing")


        # Extract Lemma CSV (using main env)
        if not extract_lemma_csv(main_env, main_docbin_path, output_csv_lemma, logger, old_id, new_id, corpus_prefix):
            overall_success = False

        # Extract UPOS CSV (using main env)
        if not extract_upos_csv(main_env, main_docbin_path, output_csv_upos, logger, old_id, new_id, corpus_prefix):
            overall_success = False

        # Extract Stopword CSV (using main env)
        if not extract_stop_csv(main_env, main_docbin_path, output_csv_stop, logger, old_id, new_id, corpus_prefix):
            overall_success = False

        # Extract Punctuation CSV (using main env)
        if not extract_dot_csv(main_env, main_docbin_path, output_csv_dot, logger, old_id, new_id, corpus_prefix):
            overall_success = False

        # Extract NER CSV (using ner env, if ner_docbin exists)
        ner_tags_available_for_conllu = False
        if ner_docbin_path:
            if not extract_ner_csv(ner_env, ner_docbin_path, output_csv_ner, logger, old_id, new_id, corpus_prefix):
                overall_success = False # Consider if NER failure should block overall success
            # Also extract NER tags to temp file for CoNLL-U
            if extract_ner_tags_to_file(ner_env, ner_docbin_path, temp_ner_tags_file, logger, old_id, new_id, corpus_prefix):
                ner_tags_available_for_conllu = True
            else:
                overall_success = False # Failed to get tags needed for CoNLL-U
                if os.path.exists(temp_ner_tags_file): # Clean up failed attempt
                     os.remove(temp_ner_tags_file)
        else:
             if logger: logger.log_operation(old_id,new_id,corpus_prefix,"", output_csv_ner,"extract","csv-ner","skipped","NER docbin not found")


        # Extract CoNLL-U (using main env, potentially reading NER tags)
        ner_tags_path_for_conllu = temp_ner_tags_file if ner_tags_available_for_conllu else None
        if not extract_conllu_file(main_env, main_docbin_path, output_conllu, new_id, ner_tags_path_for_conllu, logger, old_id, new_id, corpus_prefix):
             overall_success = False

        # Copy original text file (if found)
        if source_txt_path:
            dest_original_txt = os.path.join(texts_dir, f"{new_id}-original.txt")
            try:
                shutil.copy2(source_txt_path, dest_original_txt) # copy2 preserves metadata
                if logger:
                    logger.log_operation(
                        old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                        source_file=source_txt_path, destination_file=dest_original_txt,
                        operation_type="copy", file_type="source_txt", status="success"
                    )
            except Exception as e:
                if logger:
                    logger.log_operation(
                        old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                        source_file=source_txt_path, destination_file=dest_original_txt,
                        operation_type="copy", file_type="source_txt", status="failed",
                        details=str(e)
                    )
                # Decide if failure to copy original text is critical
                # overall_success = False

    finally:
        # --- 5. Cleanup Temporary Files ---
        for temp_file in temp_files_to_clean:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    if logger: logger.log_operation(old_id,new_id,corpus_prefix,temp_file,"","cleanup","temp_file","success")
                except OSError as e:
                    if logger: logger.log_operation(old_id,new_id,corpus_prefix,temp_file,"","cleanup","temp_file","failed", details=f"Could not remove temp file: {e}")


    if logger:
        logger.log_operation(
            old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
            source_file="", destination_file=output_dir, operation_type="process_end",
            file_type="document", status="success" if overall_success else "failed",
            details=f"Finished processing document {old_id} -> {new_id}. Overall success: {overall_success}"
        )

    return overall_success


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize corpus documents based on CSV mapping and indices, extracting various formats including CoNLL-U with NER tags.")
    parser.add_argument("--mapping-csv", required=True, help="Path to the COMMA-delimited CSV file mapping old 'document_id' to new numeric 'sort_id'.")
    parser.add_argument("--main-index-csv", required=True, help="Path to the COMMA-delimited CSV index mapping 'document_id' to main DocBin file paths ('dest_path').")
    parser.add_argument("--ner-index-csv", required=True, help="Path to the COMMA-delimited CSV index mapping 'document_id' to NER DocBin file paths ('dest_path').")
    parser.add_argument("--base-dir", required=True, help="Path to the base directory containing original source data (e.g., 'cleaned_parsed_data').")
    parser.add_argument("--output-dir", required=True, help="Path to the base directory where reorganized output will be saved.")
    parser.add_argument("--main-env", required=True, help="Name of the Conda environment containing the main linguistic model (e.g., grc_proiel_trf).")
    parser.add_argument("--ner-env", required=True, help="Name of the Conda environment containing the NER model (e.g., grc_ner_trf).")
    parser.add_argument("--log-file", default="reorganization_log.csv", help="Path for the COMMA-delimited CSV log file.")
    parser.add_argument("--wandb-project", default="corpus-reorganization", help="Name for the Weights & Biases project.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")

    args = parser.parse_args()

    start_time = time.time()

    # Initialize Logger
    logger = FileOperationLogger(
        log_file_path=args.log_file,
        use_wandb=(not args.no_wandb),
        wandb_project=args.wandb_project
    )

    print(f"Starting corpus reorganization process.")
    print(f"Mapping CSV: {args.mapping_csv}")
    print(f"Main Index CSV: {args.main_index_csv}")
    print(f"NER Index CSV: {args.ner_index_csv}")
    print(f"Base Directory: {args.base_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Main Conda Env: {args.main_env}")
    print(f"NER Conda Env: {args.ner_env}")
    print(f"Logging to CSV: {args.log_file}")
    print(f"Logging to W&B: {not args.no_wandb}")
    if not args.no_wandb:
         print(f"W&B Project: {args.wandb_project}")

    if logger.use_wandb:
         wandb.config.update(vars(args)) # Log command line args to wandb

    # Load Mappings and Indices
    print("Loading mappings and indices...")
    mappings = parse_csv_mapping(args.mapping_csv)
    main_index = load_index(args.main_index_csv)
    ner_index = load_index(args.ner_index_csv)
    print(f"Loaded {len(mappings)} ID mappings.")
    print(f"Loaded {len(main_index)} main index entries.")
    print(f"Loaded {len(ner_index)} NER index entries.")

    if not mappings or not main_index:
        print("Error: Mapping CSV or Main Index CSV could not be loaded or are empty. Exiting.")
        if logger.use_wandb: wandb.finish(exit_code=1)
        exit(1)


    # Process Documents
    processed_count = 0
    failed_count = 0
    print(f"\nProcessing {len(mappings)} documents...")

    # Wrap the loop with tqdm for a progress bar
    for old_id, new_id in tqdm.tqdm(mappings.items(), desc="Processing Documents", unit="doc"):
        processed_count += 1
        success = process_document(
            old_id=old_id,
            new_id=new_id,
            base_dir=args.base_dir,
            output_base_dir=args.output_dir,
            main_index=main_index,
            ner_index=ner_index,
            main_env=args.main_env,
            ner_env=args.ner_env,
            logger=logger
        )
        if not success:
            failed_count += 1

    # Completion Summary
    end_time = time.time()
    duration = end_time - start_time

    print("\n--- Reorganization Summary ---")
    print(f"Total documents attempted: {processed_count}")
    print(f"Successfully processed: {processed_count - failed_count}")
    print(f"Failed documents: {failed_count}")
    print(f"Total time taken: {duration:.2f} seconds")

    # Summarize and close logger (logs final summary to wandb)
    summary_stats = logger.summarize_and_close()
    print(f"Detailed log saved to: {args.log_file}")
    if logger.use_wandb:
        print(f"W&B Run URL: {wandb.run.get_url() if wandb.run else 'N/A'}")

    print("\nScript finished.")
