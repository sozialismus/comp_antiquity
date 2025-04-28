#!/usr/bin/env python3
"""
Script to reorganize corpus documents according to the database schema.
Reads document mappings from CSV and uses indices to locate docbin files.
Extracts various formats (TXT, CSVs, CoNLL-U) using specified spaCy models
in separate conda environments via temporary scripts. Includes NER tags in CoNLL-U MISC column.
Includes comprehensive logging via wandb and CSV.

CSV Delimiter Confirmation: All CSV files read or written by this script
use COMMAS (,) as delimiters. This includes the input mapping/index CSVs
and all generated CSV annotation files (using QUOTE_ALL for safety).
"""
import argparse
import csv
# import json # Not used
import os
import re
import shutil
import subprocess
import time
import tempfile
import traceback
import logging # Import standard logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import tqdm
import wandb

# --- Configure standard logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler()]
)


# --- Utility function (run_python_script_in_conda_env) ---
def run_python_script_in_conda_env(
    conda_env: str,
    script_content: str,
    log_context: Dict[str, Any],
    logger: Optional['FileOperationLogger'] = None,
    timeout: int = 300
) -> bool:
    """Writes python code to a temp file and runs it in a specified conda env."""
    temp_script_fd, temp_script_path = tempfile.mkstemp(suffix='.py', text=True)
    os.close(temp_script_fd)
    success = False
    log_file_type = log_context.get('file_type', 'unknown_script')
    try:
        with open(temp_script_path, 'w', encoding='utf-8') as f_script:
            f_script.write("#!/usr/bin/env python3\n")
            f_script.write("# -*- coding: utf-8 -*-\n")
            f_script.write(script_content)
        cmd = f"conda run -n {conda_env} python {temp_script_path}"
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=timeout, encoding='utf-8')
        stderr_output = result.stderr.strip() if result.stderr else ""
        noisy_warnings = ["FutureWarning: You are using `torch.load`"]
        if stderr_output and not any(warning in stderr_output for warning in noisy_warnings) and logger:
             logger.log_operation(
                 **log_context, operation_type="extract", status="warning",
                 details=f"Extraction script stderr: {stderr_output}"
             )
        success = True
    except subprocess.TimeoutExpired as e:
         stderr_output = e.stderr.strip() if e.stderr else "N/A"
         stdout_output = e.stdout.strip() if e.stdout else "N/A"
         error_details = f"Command timed out ({e.timeout}s): {e.cmd}\nStderr: {stderr_output}\nStdout: {stdout_output}"
         if logger: logger.log_operation(**log_context, operation_type="extract", status="failed", details=error_details)
    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.strip() if e.stderr else "N/A"
        stdout_output = e.stdout.strip() if e.stdout else "N/A"
        error_details = f"Command failed (Exit Code {e.returncode}): {e.cmd}\nStderr: {stderr_output}\nStdout: {stdout_output}"
        if logger: logger.log_operation(**log_context, operation_type="extract", status="failed", details=error_details)
    except Exception as e:
        if logger: logger.log_operation(**log_context, operation_type="extract", status="failed", details=f"Unexpected error setting up/running temp script: {type(e).__name__}: {e}")
    finally:
        if os.path.exists(temp_script_path):
            try: os.unlink(temp_script_path)
            except OSError as unlink_e:
                 if logger: logger.log_operation(
                     old_id=log_context.get('old_id','?'), new_id=log_context.get('new_id','?'), corpus_prefix=log_context.get('corpus_prefix','?'),
                     source_file=temp_script_path, destination_file="", operation_type="cleanup", file_type="temp_script", status="failed",
                     details=f"Could not remove temp script: {unlink_e}")
    if success and logger:
        logger.log_operation(**log_context, operation_type="extract", status="success")
    return success

# --- FileOperationLogger ---
class FileOperationLogger:
    """Logs operations to CSV and optionally WandB."""
    def __init__(self, log_file_path: str, use_wandb: bool = True, wandb_project: str = "corpus-reorganization"):
        self.log_file_path = log_file_path
        self.log_entries = []
        self.use_wandb = use_wandb
        self.run_name = f"reorganization-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        try:
            log_dir = os.path.dirname(log_file_path)
            if log_dir: os.makedirs(log_dir, exist_ok=True)
            with open(log_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'old_id', 'new_id', 'corpus_prefix','source_file', 'destination_file', 'operation_type','file_type', 'status', 'details'])
            logging.info(f"CSV logging initialized at: {log_file_path}")
        except OSError as e:
            logging.error(f"Error creating log file {log_file_path}: {e}. Logging disabled for CSV.")
            self.log_file_path = None
        if use_wandb:
            try:
                wandb.init(project=wandb_project, name=self.run_name)
                wandb.config.update({"log_file_intended": log_file_path,"log_file_actual": self.log_file_path if self.log_file_path else "N/A","start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
                logging.info(f"WandB logging initialized for run: {wandb.run.name} (URL: {wandb.run.get_url()})")
            except Exception as e:
                logging.error(f"Error initializing wandb: {e}. Wandb logging disabled.")
                self.use_wandb = False

    def log_operation(self, status: str = "success", details: str = "", **log_ctx) -> None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        old_id = log_ctx.get('old_id', '?')
        new_id = log_ctx.get('new_id', '?')
        corpus_prefix = log_ctx.get('corpus_prefix', '?')
        src_str = str(log_ctx.get('source_file', ''))
        dest_str = str(log_ctx.get('destination_file', ''))
        op_type = log_ctx.get('operation_type', '?')
        file_type = log_ctx.get('file_type', '?')
        details_str = str(details)
        entry = {'timestamp': timestamp,'old_id': old_id,'new_id': new_id,'corpus_prefix': corpus_prefix,'source_file': src_str,'destination_file': dest_str,'operation_type': op_type,'file_type': file_type,'status': status,'details': details_str}
        self.log_entries.append(entry)
        log_level = logging.INFO
        if status == "failed": log_level = logging.ERROR
        elif status == "warning": log_level = logging.WARNING
        log_msg = f"[{op_type}/{file_type}] ID:{old_id}->{new_id} Status:{status}"
        if src_str: log_msg += f" Src:{os.path.basename(src_str)}"
        if dest_str: log_msg += f" Dest:{os.path.basename(dest_str)}"
        if details_str: log_msg += f" Details: {details_str[:200]}{'...' if len(details_str)>200 else ''}"
        logging.log(log_level, log_msg)
        if self.log_file_path:
            try:
                with open(self.log_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([timestamp, old_id, new_id, corpus_prefix,src_str, dest_str, op_type,file_type, status, details_str])
            except OSError as e: logging.warning(f"Failed to write to CSV log {self.log_file_path}: {e}")
        if self.use_wandb and wandb.run:
            try:
                payload = {f"counts/status/{status}": 1, f"counts/operation/{op_type}": 1, f"counts/file_type/{file_type}": 1}
                if status == "failed": payload["errors/failure_count"] = 1; payload[f"errors/details/{op_type}_{file_type}"] = details_str[:500] # Log truncated details
                wandb.log(payload, commit=True)
            except Exception as e: logging.warning(f"Failed to log operation to wandb: {e}")

    def summarize_and_close(self) -> Dict[str, Any]:
        # Calculation logic... (same as before)
        total_operations = len(self.log_entries); status_counts = {'success': 0, 'failed': 0, 'warning': 0, 'info': 0, 'skipped': 0}; operation_counts = {}; file_type_counts = {}; unique_old_ids = set(); unique_new_ids = set()
        for entry in self.log_entries:
            status_counts[entry['status']] = status_counts.get(entry['status'], 0) + 1; op_type = entry['operation_type']; operation_counts[op_type] = operation_counts.get(op_type, 0) + 1; file_type = entry['file_type']; file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1
            if entry['operation_type'] == 'process_start': unique_old_ids.add(entry['old_id']); unique_new_ids.add(entry['new_id'])
        successful_operations = status_counts.get('success', 0); failed_operations = status_counts.get('failed', 0); warning_operations = status_counts.get('warning', 0); relevant_ops_for_rate = successful_operations + failed_operations; success_rate = successful_operations / relevant_ops_for_rate if relevant_ops_for_rate > 0 else 0
        summary = {'total_operations': total_operations,'successful_operations': successful_operations,'failed_operations': failed_operations,'warning_operations': warning_operations,'other_status_operations': sum(v for k, v in status_counts.items() if k not in ['success', 'failed', 'warning']),'operation_counts': operation_counts,'file_type_counts': file_type_counts,'unique_documents_processed': len(unique_old_ids),'unique_new_ids_created': len(unique_new_ids),'success_rate': success_rate,}
        if self.use_wandb and wandb.run:
            try:
                wandb.summary.update(summary); wandb.summary.update({"completion_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
                columns = ["Metric", "Value"]; data = [["Total Operations", total_operations],["Successful Ops", successful_operations],["Failed Ops", failed_operations],["Warning Ops", warning_operations],["Other Status Ops", summary['other_status_operations']],["Success Rate (Success/Fail)", f"{success_rate:.2%}"],["Unique Docs Processed", len(unique_old_ids)],["Unique New IDs Created", len(unique_new_ids)]]
                for op_type, count in sorted(operation_counts.items()): data.append([f"Op Count: {op_type}", count])
                for file_type, count in sorted(file_type_counts.items()): data.append([f"File Type Count: {file_type}", count])
                summary_table = wandb.Table(columns=columns, data=data); wandb.log({"summary_statistics_table": summary_table})
                if self.log_file_path and os.path.exists(self.log_file_path): artifact = wandb.Artifact(f"{self.run_name}-log", type="run-log"); artifact.add_file(self.log_file_path); wandb.log_artifact(artifact)
                wandb.finish()
            except Exception as e: logging.warning(f"Failed to log summary/finish wandb run: {e}")
        logging.info("Logger summarized and closed.")
        return summary


# --- Extraction Functions ---
# (extract_text_from_docbin, create_fullstop_file, extract_lemma_csv,
#  extract_upos_csv, extract_stop_csv, extract_dot_csv, extract_ner_csv,
#  extract_ner_tags_to_file have been updated previously and are correct)

def extract_text_from_docbin(conda_env: str, docbin_path: str, output_txt_path: str, logger: Optional[FileOperationLogger] = None, **log_ctx) -> bool:
    script_content = f"""
# -*- coding: utf-8 -*-
import sys, spacy, os, traceback; from spacy.tokens import DocBin
try:
    nlp = spacy.load('grc_proiel_trf')
    doc_bin = DocBin().from_disk(r'{docbin_path}')
    docs = list(doc_bin.get_docs(nlp.vocab)); assert docs, "DocBin empty"
    doc = docs[0]
    os.makedirs(os.path.dirname(r'{output_txt_path}'), exist_ok=True)
    with open(r'{output_txt_path}', 'w', encoding='utf-8') as f: f.write(doc.text)
except Exception as e: print(f"Error: {{e}}", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.exit(1)
"""
    # Log file type reflects the filename {id}-joined.txt
    log_ctx.update({'source_file': docbin_path, 'destination_file': output_txt_path, 'file_type': 'txt-joined'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger)

def create_fullstop_file(input_txt_path: str, output_txt_path: str, logger: Optional[FileOperationLogger] = None, **log_ctx) -> bool:
    log_ctx.update({'source_file': input_txt_path, 'destination_file': output_txt_path, 'file_type': 'txt-fullstop', 'operation_type': 'create'})
    try:
        if not os.path.exists(input_txt_path):
             if logger: logger.log_operation(**log_ctx, status="skipped", details="Input joined text file missing")
             return False
        with open(input_txt_path, 'r', encoding='utf-8') as f: text = f.read()
        text = re.sub(r'\.(?!\.)', '.\n', text)
        text = re.sub(r'\s+\n', '\n', text); text = re.sub(r'\n\s+', '\n', text).strip()
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        with open(output_txt_path, 'w', encoding='utf-8') as f: f.write(text)
        if logger: logger.log_operation(**log_ctx, status="success")
        return True
    except Exception as e:
        if logger: logger.log_operation(**log_ctx, status="failed", details=str(e))
        return False

def extract_lemma_csv(conda_env: str, docbin_path: str, output_csv_path: str, logger: Optional[FileOperationLogger] = None, **log_ctx) -> bool:
    script_content = f"""
# -*- coding: utf-8 -*-
import sys, csv, spacy, os, traceback; from spacy.tokens import DocBin
try:
    nlp = spacy.load('grc_proiel_trf')
    doc_bin = DocBin().from_disk(r'{docbin_path}')
    docs = list(doc_bin.get_docs(nlp.vocab)); assert docs, "DocBin empty"
    doc = docs[0]
    os.makedirs(os.path.dirname(r'{output_csv_path}'), exist_ok=True)
    with open(r'{output_csv_path}', 'w', encoding='utf-8', newline='') as cf:
        w = csv.writer(cf, quoting=csv.QUOTE_ALL); w.writerow(['ID','TOKEN','LEMMA'])
        for i,t in enumerate(doc): w.writerow([i+1, str(t.text), t.lemma_])
except Exception as e: print(f"Error: {{e}}", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.exit(1)
"""
    log_ctx.update({'source_file': docbin_path, 'destination_file': output_csv_path, 'file_type': 'csv-lemma'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger)

def extract_upos_csv(conda_env: str, docbin_path: str, output_csv_path: str, logger: Optional[FileOperationLogger] = None, **log_ctx) -> bool:
    script_content = f"""
# -*- coding: utf-8 -*-
import sys, csv, spacy, os, traceback; from spacy.tokens import DocBin
try:
    nlp = spacy.load('grc_proiel_trf')
    doc_bin = DocBin().from_disk(r'{docbin_path}')
    docs = list(doc_bin.get_docs(nlp.vocab)); assert docs, "DocBin empty"
    doc = docs[0]
    os.makedirs(os.path.dirname(r'{output_csv_path}'), exist_ok=True)
    with open(r'{output_csv_path}', 'w', encoding='utf-8', newline='') as cf:
        w = csv.writer(cf, quoting=csv.QUOTE_ALL); w.writerow(['ID','TOKEN','UPOS'])
        for i,t in enumerate(doc): w.writerow([i+1, str(t.text), t.pos_])
except Exception as e: print(f"Error: {{e}}", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.exit(1)
"""
    log_ctx.update({'source_file': docbin_path, 'destination_file': output_csv_path, 'file_type': 'csv-upos'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger)

def extract_stop_csv(conda_env: str, docbin_path: str, output_csv_path: str, logger: Optional[FileOperationLogger] = None, **log_ctx) -> bool:
    script_content = f"""
# -*- coding: utf-8 -*-
import sys, csv, spacy, os, traceback; from spacy.tokens import DocBin
try:
    nlp = spacy.load('grc_proiel_trf')
    doc_bin = DocBin().from_disk(r'{docbin_path}')
    docs = list(doc_bin.get_docs(nlp.vocab)); assert docs, "DocBin empty"
    doc = docs[0]
    os.makedirs(os.path.dirname(r'{output_csv_path}'), exist_ok=True)
    with open(r'{output_csv_path}', 'w', encoding='utf-8', newline='') as cf:
        w = csv.writer(cf, quoting=csv.QUOTE_ALL); w.writerow(['ID','TOKEN','IS_STOP'])
        for i,t in enumerate(doc): w.writerow([i+1, str(t.text), 'TRUE' if t.is_stop else 'FALSE'])
except Exception as e: print(f"Error: {{e}}", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.exit(1)
"""
    log_ctx.update({'source_file': docbin_path, 'destination_file': output_csv_path, 'file_type': 'csv-stop'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger)

def extract_ner_csv(conda_env: str, docbin_path: str, output_csv_path: str, logger: Optional[FileOperationLogger] = None, **log_ctx) -> bool:
    """Extract NER CSV using temporary script with QUOTE_ALL and debug print."""
    script_content = f"""
# -*- coding: utf-8 -*-
import sys, csv, spacy, os, traceback
from spacy.tokens import DocBin
try:
    nlp = spacy.load('grc_ner_trf') # Use NER model
    doc_bin = DocBin().from_disk(r'{docbin_path}')
    docs = list(doc_bin.get_docs(nlp.vocab)); assert docs, "DocBin empty"
    doc = docs[0]
    # --- NER DEBUG PRINT ---
    print(f"DEBUG NER: Doc ID Context: {log_ctx.get('old_id', '?')}->{log_ctx.get('new_id', '?')}", file=sys.stderr)
    print(f"DEBUG NER: Loaded doc '{docbin_path}'. Found {{len(doc.ents)}} entities.", file=sys.stderr)
    # --- END NER DEBUG PRINT ---
    os.makedirs(os.path.dirname(r'{output_csv_path}'), exist_ok=True)
    with open(r'{output_csv_path}', 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(['ID', 'TOKEN', 'NER'])
        count = 0
        for token in doc:
            count += 1
            writer.writerow([count, str(token.text), token.ent_type_ if token.ent_type_ else 'O'])
except Exception as e:
    print(f"Error in extract_ner_csv script: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
    log_ctx.update({'source_file': docbin_path, 'destination_file': output_csv_path, 'file_type': 'csv-ner'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger)

def extract_ner_tags_to_file(conda_env: str, ner_docbin_path: str, output_tags_path: str, logger: Optional[FileOperationLogger] = None, **log_ctx) -> bool:
    script_content = f"""
# -*- coding: utf-8 -*-
import sys, spacy, os, traceback; from spacy.tokens import DocBin
try:
    nlp = spacy.load('grc_ner_trf') # Use NER model
    doc_bin = DocBin().from_disk(r'{ner_docbin_path}')
    docs = list(doc_bin.get_docs(nlp.vocab)); assert docs, "DocBin empty"
    doc = docs[0]
    os.makedirs(os.path.dirname(r'{output_tags_path}'), exist_ok=True)
    with open(r'{output_tags_path}', 'w', encoding='utf-8') as f:
        f.write('\\n'.join([token.ent_type_ if token.ent_type_ else 'O' for token in doc]))
except Exception as e: print(f"Error: {{e}}", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.exit(1)
"""
    log_ctx.update({'source_file': ner_docbin_path, 'destination_file': output_tags_path, 'file_type': 'ner-tags'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger)

def extract_dot_csv(conda_env: str, docbin_path: str, output_csv_path: str, logger: Optional[FileOperationLogger] = None, **log_ctx) -> bool:
    script_content = f"""
# -*- coding: utf-8 -*-
import sys, csv, spacy, os, traceback; from spacy.tokens import DocBin
try:
    nlp = spacy.load('grc_proiel_trf')
    doc_bin = DocBin().from_disk(r'{docbin_path}')
    docs = list(doc_bin.get_docs(nlp.vocab)); assert docs, "DocBin empty"
    doc = docs[0]
    os.makedirs(os.path.dirname(r'{output_csv_path}'), exist_ok=True)
    with open(r'{output_csv_path}', 'w', encoding='utf-8', newline='') as cf:
        w = csv.writer(cf, quoting=csv.QUOTE_ALL); w.writerow(['ID','TOKEN','IS_PUNCT'])
        for i,t in enumerate(doc): w.writerow([i+1, str(t.text), 'TRUE' if t.is_punct else 'FALSE'])
except Exception as e: print(f"Error: {{e}}", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.exit(1)
"""
    log_ctx.update({'source_file': docbin_path, 'destination_file': output_csv_path, 'file_type': 'csv-dot'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger)

# --- USE YOUR UPDATED extract_conllu_file FUNCTION ---
def extract_conllu_file(
    conda_env: str, main_docbin_path: str, output_conllu_path: str, doc_id: str,
    ner_tags_path: Optional[str], logger: Optional[FileOperationLogger] = None, **log_ctx
) -> bool:
    """Extract CoNLL-U using temporary script, incorporating NER tags."""
    script_content = f"""
# -*- coding: utf-8 -*-
import sys, spacy, os, traceback
from spacy.tokens import DocBin

# --- Configuration ---
main_docbin_path = r'{main_docbin_path}'
output_conllu_path = r'{output_conllu_path}'
ner_tags_path = {repr(ner_tags_path)}
doc_id_str = '{doc_id}'
# --- End Configuration ---

try:
    nlp = spacy.load('grc_proiel_trf')
    doc_bin = DocBin().from_disk(main_docbin_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    if not docs: raise ValueError("DocBin file contains no documents.")
    doc = docs[0]

    ner_tags = None
    if ner_tags_path and os.path.exists(ner_tags_path):
        try:
            with open(ner_tags_path, 'r', encoding='utf-8') as f_ner:
                ner_tags = [line.strip() for line in f_ner if line.strip()]
            if len(ner_tags) != len(doc):
                print(f"Warning (doc {{doc_id_str}}): Mismatch token count ({{len(doc)}}) vs NER tag count ({{len(ner_tags)}}).", file=sys.stderr)
        except Exception as e:
            print(f"Warning (doc {{doc_id_str}}): Could not read NER tags from {{ner_tags_path}}: {{e}}.", file=sys.stderr)
            ner_tags = None
    elif ner_tags_path:
         print(f"Warning (doc {{doc_id_str}}): NER tags path does not exist: {{ner_tags_path}}.", file=sys.stderr)

    os.makedirs(os.path.dirname(output_conllu_path), exist_ok=True)

    with open(output_conllu_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"# newdoc id = {{doc_id_str}}\\n")
        sent_id_counter = 1
        for sent in doc.sents:
            sent_text_clean = sent.text.replace('\\n', ' ').replace('\\r', '')
            f_out.write(f"# sent_id = {{doc_id_str}}-{{sent_id_counter}}\\n")
            f_out.write(f"# text = {{sent_text_clean}}\\n")

            token_sent_id_counter = 1
            for token in sent:
                # Proper head ID calculation, with fallback to 0 for root
                head_id = token.head.i - sent.start + 1 if token.head.i != token.i else 0

                # Ensure morph features are properly formatted or set to "_"
                feats = str(token.morph) if token.morph and str(token.morph).strip() else "_"

                # Process MISC field with NER tags if available
                misc_parts = []
                if ner_tags:
                    try:
                        ner_tag = ner_tags[token.i]
                        if ner_tag and ner_tag != 'O':
                             misc_parts.append(f"NER={{ner_tag}}")
                    except IndexError:
                         pass  # Skip rather than add error marker

                # Add SpaceAfter=No if needed
                if token.i + 1 < len(doc) and doc[token.i+1].idx == token.idx + len(token.text):
                     misc_parts.append("SpaceAfter=No")

                misc_field = "|".join(misc_parts) if misc_parts else "_"

                # Always use "_" for empty DEPS field (9th column) - required by UD
                deps_field = "_"

                # Ensure valid DEPREL - fallback to "dep" if empty
                deprel = str(token.dep_) if token.dep_ and token.dep_.strip() else "dep"

                columns = [
                    str(token_sent_id_counter),      # ID
                    str(token.text),                 # FORM
                    str(token.lemma_),               # LEMMA
                    str(token.pos_),                 # UPOS
                    str(token.tag_),                 # XPOS
                    feats,                           # FEATS
                    str(head_id),                    # HEAD
                    deprel,                          # DEPREL
                    deps_field,                      # DEPS
                    misc_field                       # MISC
                ]
                f_out.write("\\t".join(columns) + "\\n")
                token_sent_id_counter += 1
            f_out.write("\\n")
            sent_id_counter += 1

except Exception as e:
    print(f"Error in extract_conllu_file script for doc {{doc_id_str}}: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
"""
    log_ctx.update({'source_file': main_docbin_path, 'destination_file': output_conllu_path, 'file_type': 'conllu'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger, timeout=600)


# --- Helper Functions ---
# (load_index, parse_csv_mapping, find_original_text remain the same)
def load_index(index_path: str) -> Dict[str, str]:
    index = {}
    try:
        df = pd.read_csv(index_path)
        required_path_col = 'processed_path'
        if 'document_id' in df.columns and required_path_col in df.columns:
            for _, row in df.iterrows():
                doc_id = str(row['document_id']).strip()
                file_path = str(row[required_path_col]).strip()
                if doc_id and file_path: index[doc_id] = file_path
        else: logging.warning(f"Index file '{index_path}' missing required columns 'document_id' or '{required_path_col}'.")
    except FileNotFoundError: logging.error(f"Index file not found at '{index_path}'"); return {}
    except Exception as e: logging.error(f"Error loading index file '{index_path}': {e}")
    return index

def parse_csv_mapping(csv_path: str) -> Dict[str, str]:
    mappings = {}
    try:
        df = pd.read_csv(csv_path)
        if 'document_id' in df.columns and 'sort_id' in df.columns:
             for index, row in df.iterrows():
                old_id = str(row['document_id']).strip()
                new_id_val = row['sort_id']
                if pd.notna(new_id_val):
                    try: new_id = str(int(new_id_val))
                    except ValueError: new_id = None; logging.warning(f"Invalid non-integer sort_id '{new_id_val}' for doc '{old_id}'. Skipping.")
                else: new_id = None
                if old_id and new_id: mappings[old_id] = new_id
                elif old_id and new_id is None and pd.isna(row['sort_id']): logging.warning(f"Missing sort_id for doc '{old_id}'. Skipping.")
        else: logging.warning(f"Mapping file '{csv_path}' missing 'document_id' or 'sort_id'.")
    except FileNotFoundError: logging.error(f"Mapping file not found at '{csv_path}'"); return {}
    except Exception as e: logging.error(f"Error parsing mapping file '{csv_path}': {e}")
    return mappings

def find_original_text(doc_id: str, base_dir: str) -> Optional[str]:
    corpus_prefix_match = re.match(r'^([a-zA-Z0-9\-_]+?)_', doc_id)
    corpus_prefix = corpus_prefix_match.group(1) if corpus_prefix_match else ""
    source_sub_dir = "cleaned_parsed_data"
    possible_paths = []
    if corpus_prefix:
        cleaned_dir = os.path.join(base_dir, source_sub_dir, corpus_prefix)
        filename_part = doc_id.replace(f"{corpus_prefix}_", "", 1)
        possible_paths.extend([os.path.join(cleaned_dir, filename_part + ".txt"), os.path.join(cleaned_dir, doc_id + ".txt")])
    possible_paths.append(os.path.join(base_dir, source_sub_dir, doc_id + ".txt"))
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.isfile(abs_path): return abs_path
    return None

# --- Main Processing Logic ---
def process_document(old_id: str, new_id: str, base_dir: str, output_base_dir: str, main_index: Dict[str, str], ner_index: Dict[str, str], main_env: str, ner_env: str, logger: Optional[FileOperationLogger] = None) -> Tuple[bool, str]:
    overall_success = True
    final_status_msg = "Success"
    temp_files_to_clean = []
    corpus_prefix_match = re.match(r'^([a-zA-Z0-9\-_]+?)_', old_id)
    corpus_prefix = corpus_prefix_match.group(1) if corpus_prefix_match else "unknown_corpus"
    log_context_base = {'old_id': old_id, 'new_id': new_id, 'corpus_prefix': corpus_prefix}

    if logger: logger.log_operation(**log_context_base, operation_type="process_start", file_type="document", status="info", details="Started")

    main_docbin_path = main_index.get(old_id)
    ner_docbin_path = ner_index.get(old_id)
    ner_docbin_path_resolved = None
    if ner_docbin_path:
        if os.path.exists(ner_docbin_path) and os.path.isfile(ner_docbin_path): ner_docbin_path_resolved = ner_docbin_path
        else:
             if logger: logger.log_operation(**log_context_base, operation_type="lookup", file_type="ner_docbin", status="warning", details="NER index path invalid/not found.")

    source_txt_path = find_original_text(old_id, base_dir)
    if not source_txt_path and logger:
        logger.log_operation(**log_context_base, operation_type="lookup", file_type="source_txt", status="warning", details="Original text file not found.")

    output_dir = os.path.join(output_base_dir, corpus_prefix, new_id)
    texts_dir = os.path.join(output_dir, "texts")
    annotations_dir = os.path.join(output_dir, "annotations")
    try:
        os.makedirs(texts_dir, exist_ok=True); os.makedirs(annotations_dir, exist_ok=True)
    except OSError as e:
         if logger: logger.log_operation(**log_context_base, operation_type="create_dir", file_type="directory", status="failed", details=f"Error: {e}")
         return False, f"Failed: Dir creation error {e}"

    # --- FILENAME CHANGE APPLIED HERE ---
    output_txt_joined = os.path.join(texts_dir, f"{new_id}-joined.txt") # Plain text output
    output_txt_fullstop = os.path.join(texts_dir, f"{new_id}-fullstop.txt") # Fullstop-separated output
    # --- END FILENAME CHANGE ---
    output_csv_lemma = os.path.join(annotations_dir, f"{new_id}-lemma.csv")
    output_csv_upos = os.path.join(annotations_dir, f"{new_id}-upos.csv")
    output_csv_stop = os.path.join(annotations_dir, f"{new_id}-stop.csv")
    output_csv_dot = os.path.join(annotations_dir, f"{new_id}-dot.csv")
    output_csv_ner = os.path.join(annotations_dir, f"{new_id}-ner.csv")
    output_conllu = os.path.join(annotations_dir, f"{new_id}-conllu.conllu")
    temp_ner_tags_file = Path(annotations_dir) / f".{new_id}_temp_ner_tags.txt"
    temp_files_to_clean.append(str(temp_ner_tags_file))

    try:
        # --- FILENAME CHANGE USED HERE ---
        if not extract_text_from_docbin(main_env, main_docbin_path, output_txt_joined, logger, **log_context_base):
            overall_success = False; final_status_msg = "Failed: txt-joined extraction"
            if logger: logger.log_operation(**log_context_base, source_file=output_txt_joined, destination_file=output_txt_fullstop, operation_type="create", file_type="txt-fullstop", status="skipped", details="Input text failed extraction.")
        elif not create_fullstop_file(output_txt_joined, output_txt_fullstop, logger, **log_context_base):
             pass # Logged inside, not critical failure
        # --- END FILENAME CHANGE ---

        if not extract_lemma_csv(main_env, main_docbin_path, output_csv_lemma, logger, **log_context_base): overall_success = False; final_status_msg = "Failed: csv-lemma extraction"
        if not extract_upos_csv(main_env, main_docbin_path, output_csv_upos, logger, **log_context_base): overall_success = False; final_status_msg = "Failed: csv-upos extraction"
        if not extract_stop_csv(main_env, main_docbin_path, output_csv_stop, logger, **log_context_base): overall_success = False; final_status_msg = "Failed: csv-stop extraction"
        if not extract_dot_csv(main_env, main_docbin_path, output_csv_dot, logger, **log_context_base): overall_success = False; final_status_msg = "Failed: csv-dot extraction"

        ner_tags_available_for_conllu = False
        if ner_docbin_path_resolved:
            if not extract_ner_csv(ner_env, ner_docbin_path_resolved, output_csv_ner, logger, **log_context_base): pass
            if extract_ner_tags_to_file(ner_env, ner_docbin_path_resolved, str(temp_ner_tags_file), logger, **log_context_base):
                ner_tags_available_for_conllu = True
            else:
                overall_success = False; final_status_msg = "Failed: ner-tags extraction"
        else:
             if logger: logger.log_operation(**log_context_base, operation_type="extract", file_type="csv-ner", status="skipped", details="NER docbin not resolved")

        ner_tags_input_path = str(temp_ner_tags_file) if ner_tags_available_for_conllu else None
        if not extract_conllu_file(main_env, main_docbin_path, output_conllu, new_id, ner_tags_input_path, logger, **log_context_base):
             overall_success = False; final_status_msg = "Failed: conllu extraction"

        if source_txt_path:
            dest_original_txt = os.path.join(texts_dir, f"{new_id}-original.txt")
            try: shutil.copy2(source_txt_path, dest_original_txt);
            except Exception as e:
                if logger: logger.log_operation(**log_context_base, operation_type="copy", file_type="source_txt", status="failed", details=str(e))

    finally:
        for temp_file_path in temp_files_to_clean:
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError as e:
                    if logger: logger.log_operation(**log_context_base, operation_type="cleanup", file_type="temp_file", status="failed", details=f"Error: {e}")

    if logger: logger.log_operation(**log_context_base, operation_type="process_end", file_type="document", status="success" if overall_success else "failed", details=f"Finished. Status: {final_status_msg}")
    return overall_success, final_status_msg


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize corpus documents...")
    # Args definition... (same)
    parser.add_argument("--mapping-csv", required=True, help="Path to the COMMA-delimited CSV file mapping old 'document_id' to new numeric 'sort_id'.")
    parser.add_argument("--main-index-csv", required=True, help="Path to the COMMA-delimited CSV index mapping 'document_id' to main DocBin file paths ('processed_path').")
    parser.add_argument("--ner-index-csv", required=True, help="Path to the COMMA-delimited CSV index mapping 'document_id' to NER DocBin file paths ('processed_path').")
    parser.add_argument("--base-dir", required=True, help="Path to the base directory containing original source data (e.g., 'cleaned_parsed_data' subdir).")
    parser.add_argument("--output-dir", required=True, help="Path to the base directory where reorganized output will be saved.")
    parser.add_argument("--main-env", required=True, help="Name of the Conda environment containing the main linguistic model (e.g., grc_proiel_trf).")
    parser.add_argument("--ner-env", required=True, help="Name of the Conda environment containing the NER model (e.g., grc_ner_trf).")
    parser.add_argument("--log-file", default="reorganization_log.csv", help="Path for the COMMA-delimited CSV log file (Default: ./reorganization_log.csv).")
    parser.add_argument("--wandb-project", default="corpus-reorganization", help="Name for the Weights & Biases project.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")

    args = parser.parse_args()

    start_time = time.time()
    logger = FileOperationLogger(log_file_path=args.log_file, use_wandb=(not args.no_wandb), wandb_project=args.wandb_project)

    logging.info(f"Starting corpus reorganization process.")
    logging.info("-" * 30)
    for arg, value in vars(args).items(): logging.info(f"{arg:<20}: {value}")
    logging.info("-" * 30)

    # --- Check for identical index files ---
    if os.path.abspath(args.main_index_csv) == os.path.abspath(args.ner_index_csv):
        logging.warning("="*60)
        logging.warning("Warning: --main-index-csv and --ner-index-csv point to the same file.")
        logging.warning("         NER extraction will likely fail or produce 'O' tags only")
        logging.warning("         if this file does not index DocBins created specifically")
        logging.warning("         with the NER model (`grc_ner_trf`).")
        logging.warning("="*60)
    # --- End Check ---

    # --- FIX: Update wandb config carefully ---
    if logger.use_wandb and wandb.run:
         try:
              args_for_config = vars(args).copy()
              current_wandb_log_file_actual = wandb.config.get("log_file_actual")
              if "log_file" in args_for_config and current_wandb_log_file_actual is not None: del args_for_config["log_file"]
              wandb.config.update(args_for_config, allow_val_change=True)
         except Exception as e: logging.warning(f"Failed to update wandb config: {e}")
    # --- END FIX ---

    logging.info("Loading mappings and indices...")
    mappings = parse_csv_mapping(args.mapping_csv)
    main_index = load_index(args.main_index_csv)
    ner_index = load_index(args.ner_index_csv)
    logging.info(f"Loaded {len(mappings)} ID mappings.")
    logging.info(f"Loaded {len(main_index)} main index entries.")
    logging.info(f"Loaded {len(ner_index)} NER index entries.")

    if not mappings or not main_index:
        logging.error("Mapping CSV or Main Index CSV could not be loaded or are empty. Exiting.")
        if logger.use_wandb and wandb.run:
             try: wandb.finish(exit_code=1)
             except Exception: pass
        exit(1)

    processed_count = 0
    failed_count = 0
    skipped_count = 0
    logging.info(f"Processing {len(mappings)} documents listed in mapping file...")

    pbar = tqdm.tqdm(mappings.items(), desc="Processing Documents", unit="doc", dynamic_ncols=True)
    for old_id, new_id in pbar:
        processed_count += 1
        main_docbin_path_check = main_index.get(old_id)
        doc_status = "Starting"
        pbar.set_postfix_str(f"ID: {old_id} -> {new_id}, Status: {doc_status}", refresh=False) # Less frequent refresh

        if not main_docbin_path_check:
             doc_status = "SKIPPED (No main index entry)"
             if logger: logger.log_operation(old_id=old_id, new_id=new_id, corpus_prefix=(old_id.split('_')[0] if '_' in old_id else 'unknown'), operation_type="lookup", file_type="main_docbin", status="skipped", details="Not found in main index.")
             skipped_count += 1
             pbar.set_postfix_str(f"ID: {old_id} -> {new_id}, Status: {doc_status}", refresh=True)
             continue
        elif not os.path.exists(main_docbin_path_check):
             doc_status = f"SKIPPED (Main DocBin not found)" # Shorten message
             if logger: logger.log_operation(old_id=old_id, new_id=new_id, corpus_prefix=(old_id.split('_')[0] if '_' in old_id else 'unknown'), operation_type="lookup", file_type="main_docbin", status="skipped", details="Path not found on disk.")
             skipped_count += 1
             pbar.set_postfix_str(f"ID: {old_id} -> {new_id}, Status: {doc_status}", refresh=True)
             continue

        success, final_status_msg = process_document(
            old_id=old_id, new_id=new_id, base_dir=args.base_dir, output_base_dir=args.output_dir,
            main_index=main_index, ner_index=ner_index, main_env=args.main_env, ner_env=args.ner_env, logger=logger
        )
        if not success:
            failed_count += 1
            doc_status = final_status_msg
        else:
            doc_status = "Success"

        pbar.set_postfix_str(f"ID: {old_id} -> {new_id}, Status: {doc_status}", refresh=True) # Refresh after processing


    # --- Completion Summary ---
    end_time = time.time()
    duration = end_time - start_time
    attempted_processing = processed_count - skipped_count

    summary_msg = "\n" + "-"*30 + \
                  "\n--- Reorganization Summary ---" + \
                  f"\nDocuments in mapping file: {processed_count}" + \
                  f"\nDocuments skipped (missing main docbin): {skipped_count}" + \
                  f"\nDocuments attempted processing: {attempted_processing}" + \
                  f"\n-> Successfully processed: {attempted_processing - failed_count}" + \
                  f"\n-> Failed processing: {failed_count}" + \
                  f"\nTotal time taken: {duration:.2f} seconds ({duration/60:.2f} minutes)" + \
                  "\n" + "-"*30
    logging.info(summary_msg)
    print(summary_msg) # Ensure summary is printed

    summary_stats = logger.summarize_and_close()
    if logger.log_file_path: logging.info(f"Detailed log saved to: {logger.log_file_path}")
    if not args.no_wandb:
        if wandb.run and wandb.run.url: # Check run exists and has URL
             logging.info(f"W&B Run URL: {wandb.run.url}")
        else: logging.info("W&B logging was enabled but run may have finished or URL not available.")

    logging.info("Script finished.")
    print("\nScript finished.")
