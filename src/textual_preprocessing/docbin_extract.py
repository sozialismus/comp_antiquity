#!/usr/bin/env python3
"""
Script to reorganize corpus documents according to the database schema.
Reads document mappings from CSV and uses indices to locate docbin files.
Extracts various formats (TXT, CSVs, CoNLL-U) using specified spaCy models
in separate conda environments via temporary scripts. Attempts NER tag alignment.
Includes NER tags in CoNLL-U MISC column and detailed mismatch logging.
Includes comprehensive logging via wandb and CSV. Enhanced with validation,
profiling, file locking (optional), and graceful shutdown.

CSV Delimiter Confirmation: All CSV files read or written by this script
use COMMAS (,) as delimiters. This includes the input mapping/index CSVs
and all generated CSV annotation files (using QUOTE_ALL for safety).
"""
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
import json # Added
import glob # Added
from collections import Counter # Added
from difflib import SequenceMatcher # Added
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError


# --- Configure standard logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
script_logger = logging.getLogger("ReorgScript")

# --- FIX 1: Define DummyFileLock unconditionally at the top level ---
class DummyFileLock:
    """A dummy class that mimics filelock.FileLock for when filelock is not available or not used."""
    def __init__(self, lock_file, timeout=-1):
        # Store lock_file path for potential debugging, but don't use it
        self._lock_file = lock_file
        self._timeout = timeout # Store timeout, though not used in dummy ops
        script_logger.debug(f"Using DummyFileLock for path: {self._lock_file}")

    def __enter__(self):
        script_logger.debug(f"DummyFileLock acquired (enter) for: {self._lock_file}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        script_logger.debug(f"DummyFileLock released (exit) for: {self._lock_file}")
        # No actual unlocking needed

    def acquire(self, timeout=None, poll_interval=0.05):
        # timeout = timeout if timeout is not None else self._timeout # Honor timeout if passed
        # Always succeed immediately in dummy version
        script_logger.debug(f"DummyFileLock acquired (acquire) for: {self._lock_file}")
        return self

    def release(self, force=False):
        # No actual unlocking needed
        script_logger.debug(f"DummyFileLock released (release) for: {self._lock_file}")
        pass

class DummyFileLockTimeout(Exception):
    """Dummy Timeout exception for consistency."""
    pass
# --- END FIX 1 Part 1 ---


# Optional dependency for file locking
try:
    import filelock
    FILELOCK_AVAILABLE = True
    # Ensure filelock.Timeout is accessible directly if needed later
    FileLockTimeout = filelock.Timeout
    script_logger.info("Real 'filelock' library found and available.")
except ImportError:
    FILELOCK_AVAILABLE = False
    logging.warning("Optional dependency 'filelock' not found. File locking disabled.")
    # --- FIX 1 Part 2: Mock the module using the globally defined DummyFileLock ---
    # Mock the filelock module interface using the Dummy classes defined above
    filelock = type('module', (), {
        'FileLock': DummyFileLock,
        'Timeout': DummyFileLockTimeout # Use the dummy timeout exception
    })()
    FileLockTimeout = DummyFileLockTimeout # Assign the dummy timeout for consistent access pattern
    # --- END FIX 1 Part 2 ---

import pandas as pd
import tqdm
import wandb


# --- (#9) Performance Profiling Decorator ---
def timed_operation(operation_name):
    """Decorator to log the execution time of a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_logger = logging.getLogger(f"Timer.{operation_name}")
            start_time = time.monotonic()
            try: result = func(*args, **kwargs); return result
            finally:
                duration = time.monotonic() - start_time; log_ctx = {}
                # Try to find log context from args or kwargs
                if args and isinstance(args[-1], dict) and 'old_id' in args[-1] and 'new_id' in args[-1]:
                     log_ctx = args[-1]
                elif kwargs.get('log_context'):
                     log_ctx = kwargs['log_context']
                elif args and isinstance(args[0], dict) and 'old_id' in args[0] and 'new_id' in args[0]: # Check first arg if it's the task dict
                     log_ctx = args[0]

                doc_info = f" (Doc: {log_ctx.get('old_id','?')}->{log_ctx.get('new_id','?')})" if log_ctx else ""
                func_logger.info(f"Finished in {duration:.3f}s{doc_info}")
                # Use try-except for wandb logging as run might end early
                try:
                    if wandb.run: wandb.log({f"timing/{operation_name}_duration_sec": duration}, commit=False)
                except Exception: pass # Ignore wandb errors during timing log
        return wrapper
    return decorator

# --- Utility function (run_python_script_in_conda_env) ---
@timed_operation("run_conda_script")
def run_python_script_in_conda_env(
    conda_env: str, script_content: str, log_context: Dict[str, Any],
    logger: Optional['FileOperationLogger'] = None, timeout: int = 300
) -> bool:
    """Writes python code to a temp file and runs it in a specified conda env."""
    temp_script_fd, temp_script_path = tempfile.mkstemp(suffix='.py', text=True)
    os.close(temp_script_fd)
    success = False
    log_file_type = log_context.get('file_type', 'unknown_script') # Extract file type here for consistent use
    try:
        with open(temp_script_path, 'w', encoding='utf-8') as f_script:
            f_script.write("#!/usr/bin/env python3\n"); f_script.write("# -*- coding: utf-8 -*-\n"); f_script.write(script_content)
        cmd = f"conda run -n {conda_env} python {temp_script_path}"
        # Use full path to conda run if necessary, or ensure conda is in PATH of executing shell
        # cmd = f"/path/to/conda/bin/conda run -n {conda_env} python {temp_script_path}"
        script_logger.debug(f"Executing command: {cmd}")
        result = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True, timeout=timeout, encoding='utf-8') # check=False initially

        stderr_output = result.stderr.strip() if result.stderr else ""
        stdout_output = result.stdout.strip() if result.stdout else "" # Capture stdout too

        # Check return code *after* capturing output
        if result.returncode != 0:
             error_details = f"Command failed (Exit Code {result.returncode}): {cmd}\nStderr: {stderr_output}\nStdout: {stdout_output}"
             if logger: logger.log_operation(**log_context, operation_type="extract", file_type=log_file_type, status="failed", details=error_details)
             # Raise an exception to be caught below if needed, or just return False
             # raise subprocess.CalledProcessError(result.returncode, cmd, output=stdout_output, stderr=stderr_output)
             success = False # Explicitly set success to False
        else:
            success = True # Command succeeded

        # Log warnings from stderr even on success, unless noisy
        noisy_warnings = ["FutureWarning: You are using `torch.load`"]
        if stderr_output and not any(warning in stderr_output for warning in noisy_warnings) and not stderr_output.startswith("DEBUG") and logger:
             logger.log_operation(**log_context, operation_type="extract", file_type=log_file_type, status="warning", details=f"Extraction script stderr: {stderr_output}")

    except subprocess.TimeoutExpired as e:
         stderr_output = e.stderr.strip() if e.stderr else "N/A"; stdout_output = e.stdout.strip() if e.stdout else "N/A"
         error_details = f"Command timed out ({e.timeout}s): {e.cmd}\nStderr: {stderr_output}\nStdout: {stdout_output}"
         if logger: logger.log_operation(**log_context, operation_type="extract", file_type=log_file_type, status="failed", details=error_details)
         success = False # Timeout is failure
    except Exception as e:
        if logger: logger.log_operation(**log_context, operation_type="extract", file_type=log_file_type, status="failed", details=f"Unexpected error setting up/running temp script: {type(e).__name__}: {e}")
        success = False # Other exceptions are failures
    finally:
        if os.path.exists(temp_script_path):
            try: os.unlink(temp_script_path)
            except OSError as unlink_e:
                 cleanup_log_ctx = {k: v for k, v in log_context.items() if k in ['old_id', 'new_id', 'corpus_prefix']}
                 if logger: logger.log_operation(**cleanup_log_ctx, source_file=temp_script_path, destination_file="", operation_type="cleanup", file_type="temp_script", status="failed", details=f"Could not remove temp script: {unlink_e}")

    # Log final success status (only if no failure occurred and no warning was logged)
    if success and logger:
        if not (stderr_output and not any(warning in stderr_output for warning in noisy_warnings) and not stderr_output.startswith("DEBUG")):
             logger.log_operation(**log_context, operation_type="extract", file_type=log_file_type, status="success")
    return success


# --- FileOperationLogger ---
class FileOperationLogger:
    """Logs operations to CSV and optionally WandB."""
    def __init__(self, log_file_path: str, use_wandb: bool = True, wandb_project: str = "corpus-reorganization"):
        self.log_file_path = log_file_path; self.log_entries = []; self.use_wandb = use_wandb; self.run_name = f"reorganization-{datetime.now().strftime('%Y%m%d-%H%M%S')}"; self.logger = logging.getLogger("FileLogger")
        self.wandb_run_url = None # Store URL if init succeeds
        try:
            log_dir = os.path.dirname(log_file_path);
            if log_dir: os.makedirs(log_dir, exist_ok=True)
            with open(log_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['timestamp', 'old_id', 'new_id', 'corpus_prefix','source_file', 'destination_file', 'operation_type','file_type', 'status', 'details'])
            self.logger.info(f"CSV logging initialized at: {log_file_path}")
        except OSError as e: self.logger.error(f"Error creating log file {log_file_path}: {e}. Logging disabled for CSV."); self.log_file_path = None

        if use_wandb:
            try:
                # Ensure wandb is imported before use
                import wandb
                wandb.init(project=wandb_project, name=self.run_name);
                wandb.config.update({
                    "log_file_intended": log_file_path,
                    "log_file_actual": self.log_file_path if self.log_file_path else "N/A",
                    "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                self.wandb_run_url = wandb.run.url if wandb.run else None
                self.logger.info(f"WandB logging initialized for run: {wandb.run.name} (URL: {self.wandb_run_url or 'N/A'})")
            except ImportError:
                 self.logger.error("WandB logging requires the 'wandb' library. Install it (`pip install wandb`) and log in (`wandb login`). Wandb disabled.")
                 self.use_wandb = False
            except Exception as e:
                 self.logger.error(f"Error initializing wandb: {e}. Wandb logging disabled."); self.use_wandb = False

    def log_operation(self, status: str = "success", details: str = "", **log_ctx) -> None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S'); old_id = log_ctx.get('old_id', '?'); new_id = log_ctx.get('new_id', '?'); corpus_prefix = log_ctx.get('corpus_prefix', '?'); src_str = str(log_ctx.get('source_file', '')); dest_str = str(log_ctx.get('destination_file', '')); op_type = log_ctx.get('operation_type', '?'); file_type = log_ctx.get('file_type', '?'); details_str = str(details)
        entry = {'timestamp': timestamp,'old_id': old_id,'new_id': new_id,'corpus_prefix': corpus_prefix,'source_file': src_str,'destination_file': dest_str,'operation_type': op_type,'file_type': file_type,'status': status,'details': details_str}; self.log_entries.append(entry)
        log_level = logging.INFO;
        if status == "failed": log_level = logging.ERROR
        elif status == "warning": log_level = logging.WARNING
        msg_parts = [f"[{op_type}/{file_type}]", f"ID:{old_id}->{new_id}", f"Status:{status}"]
        if src_str and src_str != 'multiple' and src_str != '?': msg_parts.append(f"Src:{os.path.basename(src_str)}")
        if dest_str and dest_str != 'multiple' and dest_str != '?': msg_parts.append(f"Dest:{os.path.basename(dest_str)}")
        if details_str: msg_parts.append(f"Details: {details_str[:200]}{'...' if len(details_str)>200 else ''}")
        log_msg = " ".join(msg_parts)
        self.logger.log(log_level, log_msg)

        if self.log_file_path:
            try:
                with open(self.log_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([timestamp, old_id, new_id, corpus_prefix,src_str, dest_str, op_type,file_type, status, details_str])
            except OSError as e: self.logger.warning(f"Failed to write to CSV log {self.log_file_path}: {e}")

        # Check use_wandb and if wandb.run is active
        if self.use_wandb and wandb.run:
            try:
                # Use wandb.log for incremental updates (safer for parallel)
                payload = {}
                payload[f"counts/status/{status}"] = 1
                payload[f"counts/operation/{op_type}"] = 1
                payload[f"counts/file_type/{file_type}"] = 1

                if status == "failed":
                    payload["errors/failure_count"] = 1
                    # Truncate details more aggressively for keys if needed
                    error_key = f"errors/details/{op_type}_{file_type}_{old_id[:20]}_{new_id[:10]}"
                    payload[error_key] = details_str[:300]

                wandb.log(payload, commit=False) # Commit less frequently if performance is issue
            except Exception as e: self.logger.warning(f"Failed to log operation step to wandb: {e}")

    @timed_operation("summarize_logs")
    def summarize_and_close(self) -> Dict[str, Any]:
        total_operations = len(self.log_entries);
        status_counts = Counter(entry['status'] for entry in self.log_entries)
        operation_counts = Counter(entry['operation_type'] for entry in self.log_entries)
        file_type_counts = Counter(entry['file_type'] for entry in self.log_entries)
        processed_ids = {(entry['old_id'], entry['new_id']) for entry in self.log_entries if entry['operation_type'] == 'process_start'}
        unique_old_ids = {old for old, new in processed_ids if old != '?'}
        unique_new_ids = {new for old, new in processed_ids if new != '?'}

        successful_ops_final = status_counts.get('success', 0)
        failed_ops_final = status_counts.get('failed', 0)
        warning_ops_final = status_counts.get('warning', 0)
        relevant_ops_for_rate = successful_ops_final + failed_ops_final
        success_rate = successful_ops_final / relevant_ops_for_rate if relevant_ops_for_rate > 0 else 0.0

        summary = {
            'total_logged_operations': total_operations,
            'successful_operations': successful_ops_final,
            'failed_operations': failed_ops_final,
            'warning_operations': warning_ops_final,
            'other_status_operations': sum(v for k, v in status_counts.items() if k not in ['success', 'failed', 'warning']),
            'operation_counts': dict(operation_counts),
            'file_type_counts': dict(file_type_counts),
            'unique_documents_processed': len(unique_old_ids),
            'unique_new_ids_created': len(unique_new_ids),
            'success_rate_final': success_rate, # Store the raw float
            'wandb_run_url': self.wandb_run_url # Include URL in summary data
        }

        if self.use_wandb and wandb.run:
            try:
                # Update specific summary fields instead
                wandb.summary['total_logged_operations'] = total_operations
                wandb.summary['successful_operations_final'] = successful_ops_final
                wandb.summary['failed_operations_final'] = failed_ops_final
                wandb.summary['warning_operations_final'] = warning_ops_final
                wandb.summary['unique_documents_processed'] = len(unique_old_ids)
                wandb.summary['unique_new_ids_created'] = len(unique_new_ids)
                wandb.summary['success_rate_final'] = success_rate
                wandb.summary["completion_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Log a summary table
                columns = ["Metric", "Value"];
                data = [
                    ["Total Logged Ops", total_operations],
                    ["Successful Ops (Final)", successful_ops_final],
                    ["Failed Ops (Final)", failed_ops_final],
                    ["Warning Ops (Final)", warning_ops_final],
                    ["Other Status Ops", summary['other_status_operations']],
                    ["Success Rate (Success/Fail)", round(success_rate, 4)],
                    ["Unique Docs Processed", len(unique_old_ids)],
                    ["Unique New IDs Created", len(unique_new_ids)]
                ]
                for op_type, count in sorted(operation_counts.items()): data.append([f"Op Count: {op_type}", count])
                for file_type, count in sorted(file_type_counts.items()): data.append([f"File Type Count: {file_type}", count])

                # Check for potential type issues before logging table
                for i, row in enumerate(data):
                    if not isinstance(row[1], (int, float, str, type(None))):
                         self.logger.warning(f"WandB Table: Invalid type '{type(row[1])}' for value '{row[1]}' in row {i} ('{row[0]}'). Skipping table log.")
                         data = None # Prevent logging bad table
                         break

                if data:
                    summary_table = wandb.Table(columns=columns, data=data)
                    wandb.log({"summary_statistics_table": summary_table}, commit=True) # Commit final logs

                # Log CSV artifact
                if self.log_file_path and os.path.exists(self.log_file_path):
                    artifact = wandb.Artifact(f"{self.run_name}-log", type="run-log")
                    artifact.add_file(self.log_file_path)
                    wandb.log_artifact(artifact)
                else:
                    self.logger.warning("CSV log file not found or not specified, cannot upload artifact.")

                wandb.finish()
            except Exception as e:
                 self.logger.warning(f"Failed during final WandB summary/logging/finish: {type(e).__name__}: {e}", exc_info=True)
                 try:
                     if wandb.run: wandb.finish(exit_code=1)
                 except Exception as finish_e:
                      self.logger.error(f"Failed to even finish wandb run: {finish_e}")
        else:
             self.logger.info("WandB logging was disabled or run ended prematurely.")

        self.logger.info("Logger summarized and closed.")
        return summary


# --- Extraction Functions ---
# Integrates your enhanced main model extraction logic
@timed_operation("extract_main_outputs")
def extract_main_model_outputs(
    conda_env: str, docbin_path: str, output_txt_joined: str, output_txt_fullstop: str,
    output_csv_lemma: str, output_csv_upos: str, output_csv_stop: str, output_csv_dot: str,
    output_conllu: str, ner_tags_path: Optional[str], doc_id_str: str,
    logger: Optional['FileOperationLogger'] = None, **log_ctx # Pass full context
) -> bool:
    """Loads main model once and extracts all related outputs via one temp script, attempting NER alignment."""
    # Ensure all paths are absolute for the script context
    docbin_path_abs = os.path.abspath(docbin_path)
    output_txt_joined_abs = os.path.abspath(output_txt_joined)
    output_txt_fullstop_abs = os.path.abspath(output_txt_fullstop)
    output_csv_lemma_abs = os.path.abspath(output_csv_lemma)
    output_csv_upos_abs = os.path.abspath(output_csv_upos)
    output_csv_stop_abs = os.path.abspath(output_csv_stop)
    output_csv_dot_abs = os.path.abspath(output_csv_dot)
    output_conllu_abs = os.path.abspath(output_conllu)
    ner_tags_path_abs = os.path.abspath(ner_tags_path) if ner_tags_path else None

    # Correctly escape paths for the f-string, especially on Windows
    docbin_path_esc = docbin_path_abs.replace('\\', '\\\\')
    output_txt_joined_esc = output_txt_joined_abs.replace('\\', '\\\\')
    output_txt_fullstop_esc = output_txt_fullstop_abs.replace('\\', '\\\\')
    output_csv_lemma_esc = output_csv_lemma_abs.replace('\\', '\\\\')
    output_csv_upos_esc = output_csv_upos_abs.replace('\\', '\\\\')
    output_csv_stop_esc = output_csv_stop_abs.replace('\\', '\\\\')
    output_csv_dot_esc = output_csv_dot_abs.replace('\\', '\\\\')
    output_conllu_esc = output_conllu_abs.replace('\\', '\\\\')
    ner_tags_path_repr = repr(ner_tags_path_abs.replace('\\', '\\\\')) if ner_tags_path_abs else 'None'

    # --- FIX: Define Regex patterns OUTSIDE the f-string ---
    # Define the raw patterns and replacements needed inside the script
    regex_pattern1_raw = r'\\.(?!\\.)'
    regex_replace1_raw = r'.\n'
    regex_pattern2_raw = r'\s+\n'
    regex_replace2_raw = r'\n'
    regex_pattern3_raw = r'\n\s+'
    regex_replace3_raw = r'\n'

    # Escape these patterns FOR INCLUSION as string literals *within* the target script's f-string
    # Need to double-escape backslashes and escape newlines
    def escape_for_script_string(s: str) -> str:
        return s.replace('\\', '\\\\').replace('\n', '\\n').replace("'", "\\'") # Also escape single quotes

    regex_p1_script_literal = escape_for_script_string(regex_pattern1_raw)
    regex_r1_script_literal = escape_for_script_string(regex_replace1_raw)
    regex_p2_script_literal = escape_for_script_string(regex_pattern2_raw)
    regex_r2_script_literal = escape_for_script_string(regex_replace2_raw)
    regex_p3_script_literal = escape_for_script_string(regex_pattern3_raw)
    regex_r3_script_literal = escape_for_script_string(regex_replace3_raw)
    # --- END FIX ---

    # Construct the script content, passing the escaped patterns
    script_content = f"""# -*- coding: utf-8 -*-
import sys, csv, spacy, os, traceback, re, json
from difflib import SequenceMatcher
from spacy.tokens import DocBin
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] TempScript: %(message)s', stream=sys.stderr)

# --- Paths passed from main script ---
docbin_path=r'{docbin_path_esc}'
output_txt_joined=r'{output_txt_joined_esc}'
output_txt_fullstop=r'{output_txt_fullstop_esc}'
output_csv_lemma=r'{output_csv_lemma_esc}'
output_csv_upos=r'{output_csv_upos_esc}'
output_csv_stop=r'{output_csv_stop_esc}'
output_csv_dot=r'{output_csv_dot_esc}'
output_conllu=r'{output_conllu_esc}'
ner_tags_path={ner_tags_path_repr}
doc_id_str='{doc_id_str}'

logging.info(f"Starting extraction for doc ID: {{doc_id_str}}")
logging.info(f"Main DocBin Path: {{docbin_path}}")
logging.info(f"NER Tags Path: {{ner_tags_path}}")
logging.info(f"CoNLL-U Output Path: {{output_conllu}}")

# --- Alignment Function ---
def attempt_ner_alignment(tokens, ner_tags):
    # (Keep the alignment function logic as is)
    logging.info(f"Entering attempt_ner_alignment. len(tokens)={{len(tokens)}}, len(ner_tags)={{len(ner_tags)}}")
    token_texts = [str(t.text) for t in tokens]
    logging.debug(f"Token texts: {{token_texts[:10]}}...")
    matcher = SequenceMatcher(None, token_texts, ner_tags, autojunk=False)
    aligned_tags = [None] * len(tokens)
    mismatch_details = []
    alignment_stats = {{
        'status': 'failed_no_matches', 'aligned_count': 0,
        'total_tokens': len(tokens), 'total_ner_tags': len(ner_tags),
        'success_rate': 0.0, 'details': mismatch_details
    }}
    blocks_processed = 0
    for block in matcher.get_matching_blocks():
        if block.size == 0: continue
        blocks_processed += 1
        token_start, ner_start, size = block.a, block.b, block.size
        logging.debug(f"Match block: token_idx={{token_start}}, ner_idx={{ner_start}}, size={{size}}")
        for i in range(size):
            token_idx = token_start + i; ner_idx = ner_start + i
            if token_idx < len(aligned_tags) and ner_idx < len(ner_tags):
                 aligned_tags[token_idx] = ner_tags[ner_idx]
                 alignment_stats['aligned_count'] += 1
                 if len(mismatch_details) < 20:
                     token_text_safe = token_texts[token_idx] if token_idx < len(token_texts) else "TOKEN_OOB"
                     ner_tag_safe = ner_tags[ner_idx]
                     mismatch_details.append(f"Align: tok[{{token_idx}}]='{token_text_safe}' <-> tag[{{ner_idx}}]='{ner_tag_safe}'")
            else:
                 logging.warning(f"Alignment index out of bounds: token_idx={{token_idx}} (max={{len(aligned_tags)-1}}), ner_idx={{ner_idx}} (max={{len(ner_tags)-1}})")
    logging.info(f"Alignment processed {{blocks_processed}} matching blocks. Total aligned: {{alignment_stats['aligned_count']}}")
    if alignment_stats['aligned_count'] > 0:
         alignment_stats['success_rate'] = alignment_stats['aligned_count'] / len(tokens) if len(tokens) > 0 else 0
         alignment_stats['status'] = 'success' if alignment_stats['aligned_count'] == len(tokens) else 'partial'
         if alignment_stats['status'] == 'partial':
             unaligned_tokens = len(tokens) - alignment_stats['aligned_count']
             logging.info(f"{{unaligned_tokens}} tokens could not be aligned.")
    success_threshold = 0.5
    if alignment_stats['success_rate'] < success_threshold and alignment_stats['status'] != 'success':
        logging.warning(f"Alignment quality low ({{alignment_stats['success_rate']:.1%}} < {success_threshold*100:.0f}%). Discarding alignment results.")
        alignment_stats['status'] = 'failed_low_quality'
        mismatch_details.append(f"Failed: Alignment rate below threshold ({success_threshold:.1%})")
        return None, alignment_stats
    logging.info(f"Alignment Result: Status={{alignment_stats['status']}}, Rate={{alignment_stats['success_rate']:.2%}} (Aligned={{alignment_stats['aligned_count']}}/{{len(tokens)}})")
    return aligned_tags, alignment_stats
# --- End Alignment Function ---

# Ensure output directories exist
# (Keep directory creation logic as is)
dirs_to_check=set([os.path.dirname(p) for p in [output_txt_joined,output_txt_fullstop,output_csv_lemma,output_csv_upos,output_csv_stop,output_csv_dot,output_conllu] if p])
for d in dirs_to_check:
    if d:
        try:
            os.makedirs(d, exist_ok=True); logging.info(f"Ensured directory exists: {{d}}")
        except OSError as e:
            logging.error(f"Failed to create directory {{d}}: {{e}}"); sys.exit(1)

try:
    # --- Load Model and DocBin ---
    # (Keep model loading logic as is)
    logging.info(f"Loading spaCy model 'grc_proiel_trf'...")
    nlp = spacy.load('grc_proiel_trf');
    logging.info(f"Loading DocBin from {{docbin_path}}...")
    doc_bin = DocBin().from_disk(docbin_path);
    docs = list(doc_bin.get_docs(nlp.vocab));
    if not docs:
        logging.error(f"DocBin file is empty or failed to load: {{docbin_path}}")
        sys.exit(1)
    doc = docs[0]; num_tokens = len(doc)
    logging.info(f"Successfully loaded doc with {{num_tokens}} tokens.")

    # --- Extract TXT formats ---
    doc_text = doc.text
    logging.info(f"Writing joined text to {{output_txt_joined}}...")
    with open(output_txt_joined,'w',encoding='utf-8') as f: f.write(doc_text)

    logging.info(f"Processing and writing fullstop text to {{output_txt_fullstop}}...")
    try:
        # --- FIX: Use the passed regex patterns ---
        # Define the patterns USING the escaped literals passed into the script
        # Use r'...' to ensure backslashes are treated literally within re.sub
        pat1 = r'{regex_p1_script_literal}'
        rep1 = r'{regex_r1_script_literal}'
        pat2 = r'{regex_p2_script_literal}'
        rep2 = r'{regex_r2_script_literal}'
        pat3 = r'{regex_p3_script_literal}'
        rep3 = r'{regex_r3_script_literal}'

        logging.debug(f"Applying regex: {{pat1}} -> {{rep1}}")
        tfs = re.sub(pat1, rep1, doc_text)
        logging.debug(f"Applying regex: {{pat2}} -> {{rep2}}")
        tfs = re.sub(pat2, rep2, tfs)
        logging.debug(f"Applying regex: {{pat3}} -> {{rep3}}")
        tfs = re.sub(pat3, rep3, tfs).strip()
        # --- END FIX ---

        with open(output_txt_fullstop,'w',encoding='utf-8') as f: f.write(tfs)
    except Exception as fs_e:
        logging.warning(f"Failed to generate fullstop format: {{fs_e}}", exc_info=True)

    # --- Extract CSV formats ---
    # (Keep CSV writing logic as is)
    logging.info("Writing CSV annotation files...")
    try:
        with open(output_csv_lemma,'w',encoding='utf-8',newline='') as fl, \\
             open(output_csv_upos,'w',encoding='utf-8',newline='') as fu, \\
             open(output_csv_stop,'w',encoding='utf-8',newline='') as fs, \\
             open(output_csv_dot,'w',encoding='utf-8',newline='') as fd:
            wl=csv.writer(fl,quoting=csv.QUOTE_ALL); wl.writerow(['ID','TOKEN','LEMMA']);
            wu=csv.writer(fu,quoting=csv.QUOTE_ALL); wu.writerow(['ID','TOKEN','UPOS']);
            ws=csv.writer(fs,quoting=csv.QUOTE_ALL); ws.writerow(['ID','TOKEN','IS_STOP']);
            wd=csv.writer(fd,quoting=csv.QUOTE_ALL); wd.writerow(['ID','TOKEN','IS_PUNCT'])
            for i,t in enumerate(doc):
                tid,ttxt=i+1,str(t.text);
                wl.writerow([tid,ttxt,t.lemma_]);
                wu.writerow([tid,ttxt,t.pos_]);
                ws.writerow([tid,ttxt,'TRUE' if t.is_stop else 'FALSE']);
                wd.writerow([tid,ttxt,'TRUE' if t.is_punct else 'FALSE'])
        logging.info("Finished writing CSV files.")
    except Exception as csv_e:
        logging.error(f"Failed during CSV writing: {{csv_e}}", exc_info=True); sys.exit(1)


    # --- NER Processing & Alignment ---
    # (Keep NER processing and alignment logic as is)
    original_ner_tags = None; ner_tags_to_use = None; alignment_info = None; mismatch_detected = False
    mismatch_data_for_json = {{}}
    logging.info(f"Checking for NER tags at: {{ner_tags_path}}")
    if ner_tags_path and os.path.exists(ner_tags_path):
        try:
            logging.info(f"Reading NER tags from {{ner_tags_path}}")
            with open(ner_tags_path,'r',encoding='utf-8') as fn: original_ner_tags=[ln.strip() for ln in fn if ln.strip()]
            num_ner_tags = len(original_ner_tags); logging.info(f"Read {{num_ner_tags}} NER tags.")
            mismatch_data_for_json = {{"document_id": doc_id_str,"main_model_tokens": num_tokens,"ner_model_tags": num_ner_tags,"mismatch_detected": False,"alignment_info": None}}
            if num_ner_tags != num_tokens:
                mismatch_detected = True; mismatch_data_for_json["mismatch_detected"] = True
                logging.warning(f"Token count mismatch! Main model: {{num_tokens}}, NER tags: {{num_ner_tags}}. Attempting alignment.")
                aligned_result, alignment_info = attempt_ner_alignment(doc, original_ner_tags)
                mismatch_data_for_json["alignment_info"] = alignment_info
                if aligned_result: ner_tags_to_use = aligned_result; logging.info("Using ALIGNED NER tags for CoNLL-U.")
                else: ner_tags_to_use = None; logging.warning("Alignment failed or discarded. NER tags will be OMITTED from CoNLL-U.")
            else:
                ner_tags_to_use = original_ner_tags; logging.info("Token counts match. Using original NER tags."); mismatch_data_for_json = None
        except Exception as e:
            logging.warning(f"Failed to read or align NER tags: {{e}}.", exc_info=True); ner_tags_to_use=None
            mismatch_data_for_json["error_during_ner_processing"] = str(e)
    elif ner_tags_path:
        logging.warning(f"NER tags path provided but not found: {{ner_tags_path}}")
        mismatch_data_for_json = {{ "document_id": doc_id_str, "error_message": "NER tags path not found", "path_checked": ner_tags_path }}
    else:
        logging.info("No NER tags path provided. Skipping NER integration."); mismatch_data_for_json = None

    # --- Write CoNLL-U ---
    # (Keep CoNLL-U writing logic as is)
    logging.info(f"Writing CoNLL-U file to {{output_conllu}}...")
    try:
        with open(output_conllu,"w",encoding="utf-8") as fo:
            fo.write(f"# newdoc id = {{doc_id_str}}\\n")
            if mismatch_detected:
                fo.write(f"# ner_token_mismatch = True\\n"); fo.write(f"# main_model_tokens = {{num_tokens}}\\n"); fo.write(f"# ner_model_tags = {{num_ner_tags}}\\n")
                if alignment_info:
                    fo.write(f"# ner_alignment_status = {{alignment_info.get('status','?')}}\\n"); fo.write(f"# ner_alignment_rate = {{alignment_info.get('success_rate', 0.0):.4f}}\\n")
                    if alignment_info.get('status') in ('partial', 'failed_low_quality') and alignment_info.get('details'):
                         details_preview = "; ".join(alignment_info['details'][:3]); fo.write(f"# ner_alignment_details_preview = {details_preview}\\n")
            sidc = 1
            for sent_idx, sent in enumerate(doc.sents):
                stc=str(sent.text).replace('\\n',' ').replace('\\r','').strip()
                fo.write(f"\\n# sent_id = {{doc_id_str}}-{{sidc}}\\n"); fo.write(f"# text = {{stc}}\\n")
                tsic = 1
                for token_idx_in_doc, t in enumerate(sent):
                    head_idx_in_doc = t.head.i; head_idx_in_sent = head_idx_in_doc - sent.start + 1 if head_idx_in_doc != t.i else 0
                    fts = str(t.morph) if t.morph else "_"
                    mp = []
                    if ner_tags_to_use:
                        abs_token_index = t.i
                        if abs_token_index < len(ner_tags_to_use):
                            nt = ner_tags_to_use[abs_token_index];
                            if nt and nt != 'O': mp.append(f"NER={{nt}}")
                        else: logging.warning(f"Token index {{abs_token_index}} out of bounds for ner_tags_to_use (len={{len(ner_tags_to_use)}}) in sentence {{sidc}}")
                    if (t.i + 1) < len(doc) and doc[t.i + 1].idx == (t.idx + len(t.text)): mp.append("SpaceAfter=No")
                    mf = "|".join(mp) if mp else "_"; dpr = str(t.dep_).strip() if t.dep_ else "dep";
                    if not dpr: dpr = "dep"; dpf = "_"
                    cols = [str(tsic),str(t.text),str(t.lemma_),str(t.pos_),str(t.tag_),fts,str(head_idx_in_sent),dpr,dpf,mf]
                    fo.write("\\t".join(cols) + "\\n"); tsic += 1
                sidc += 1
        logging.info("Finished writing CoNLL-U file.")
    except Exception as conllu_e:
        logging.error(f"Failed during CoNLL-U writing: {{conllu_e}}", exc_info=True); sys.exit(1)

    # --- Write mismatch details JSON ---
    # (Keep mismatch JSON writing logic as is)
    if mismatch_detected and mismatch_data_for_json:
        mismatch_info_file_path = os.path.join(os.path.dirname(output_conllu), f"{{doc_id_str}}_ner_mismatch_info.json")
        mismatch_info_file_path_esc = mismatch_info_file_path.replace('\\\\', '\\\\\\\\')
        logging.info(f"Writing mismatch info to {{mismatch_info_file_path_esc}}...")
        try:
            with open(mismatch_info_file_path, 'w', encoding='utf-8') as fm: json.dump(mismatch_data_for_json, fm, indent=2)
            logging.info("Finished writing mismatch info.")
        except Exception as e: logging.error(f"Failed to write mismatch info file: {{e}}", exc_info=True)
    elif mismatch_data_for_json and "error_message" in mismatch_data_for_json:
         mismatch_info_file_path = os.path.join(os.path.dirname(output_conllu), f"{{doc_id_str}}_ner_error_info.json")
         mismatch_info_file_path_esc = mismatch_info_file_path.replace('\\\\', '\\\\\\\\')
         logging.warning(f"Writing NER processing error info to {{mismatch_info_file_path_esc}}...")
         try:
             with open(mismatch_info_file_path, 'w', encoding='utf-8') as fm: json.dump(mismatch_data_for_json, fm, indent=2)
         except Exception as e: logging.error(f"Failed to write NER error info file: {{e}}", exc_info=True)

except Exception as e:
    logging.error(f"Unhandled exception in temporary script: {{e}}", exc_info=True)
    sys.exit(1)

logging.info(f"Extraction script finished successfully for doc ID: {{doc_id_str}}.")
sys.exit(0)
"""
    # Call the conda run function with the generated script
    run_log_ctx = log_ctx.copy()
    run_log_ctx.update({
        'source_file': docbin_path,
        'destination_file': f"Multiple files in {os.path.dirname(output_txt_joined_abs)}", # Use abs path base
        'file_type': 'main-model-outputs'
    })
    return run_python_script_in_conda_env(conda_env, script_content, run_log_ctx, logger, timeout=1200)


# Integrates your enhanced NER model extraction
@timed_operation("extract_ner_outputs")
def extract_ner_model_outputs(
    conda_env: str, docbin_path: str, output_csv_ner: str, output_ner_tags: str,
    logger: Optional[FileOperationLogger] = None, **log_ctx # Pass full context
) -> bool:
    """Loads NER model once and extracts NER CSV and NER tags file via one temp script."""
    ner_model_name = 'grc_ner_trf' # Make sure this is the correct model name

    docbin_path_abs = os.path.abspath(docbin_path)
    output_csv_ner_abs = os.path.abspath(output_csv_ner)
    output_ner_tags_abs = os.path.abspath(output_ner_tags)

    docbin_path_esc = docbin_path_abs.replace('\\', '\\\\')
    output_csv_ner_esc = output_csv_ner_abs.replace('\\', '\\\\')
    output_ner_tags_esc = output_ner_tags_abs.replace('\\', '\\\\')

    old_id_str = log_ctx.get('old_id', '?')
    new_id_str = log_ctx.get('new_id', '?')
    ner_summary_file_path = os.path.join(os.path.dirname(output_csv_ner_abs), f"{new_id_str}_ner_stats.json")
    ner_summary_file_path_esc = ner_summary_file_path.replace('\\', '\\\\')


    script_content = f"""# -*- coding: utf-8 -*-
import sys, csv, spacy, os, traceback, json
from spacy.tokens import DocBin
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] TempScript-NER: %(message)s', stream=sys.stderr)

docbin_path=r'{docbin_path_esc}'
output_csv_ner=r'{output_csv_ner_esc}'
output_ner_tags=r'{output_ner_tags_esc}'
ner_model_name='{ner_model_name}'
ner_summary_file=r'{ner_summary_file_path_esc}'
old_id_str='{old_id_str}'
new_id_str='{new_id_str}'

logging.info(f"Starting NER extraction for Doc ID: {{old_id_str}} -> {{new_id_str}}")
logging.info(f"NER DocBin Path: {{docbin_path}}")
logging.info(f"NER Model: {{ner_model_name}}")
logging.info(f"Output CSV: {{output_csv_ner}}")
logging.info(f"Output Tags: {{output_ner_tags}}")
logging.info(f"Output Summary: {{ner_summary_file}}")


dirs_to_check=set([os.path.dirname(p) for p in [output_csv_ner, output_ner_tags, ner_summary_file] if p])
for d in dirs_to_check:
    if d:
        try:
             os.makedirs(d, exist_ok=True); logging.info(f"Ensured directory exists: {{d}}")
        except OSError as e: logging.error(f"Failed create dir {{d}}: {{e}}"); sys.exit(1)

try:
    logging.info(f"Loading NER model '{{ner_model_name}}'...")
    nlp = spacy.load(ner_model_name)
    logging.info(f"Loading DocBin from {{docbin_path}}...")
    doc_bin = DocBin().from_disk(docbin_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    if not docs:
        logging.error(f"NER DocBin is empty or failed load: {{docbin_path}}")
        sys.exit(1)
    doc = docs[0]; num_tokens = len(doc)
    logging.info(f"Loaded doc with {{num_tokens}} tokens. Found {{len(doc.ents)}} entities.")

    ner_stats={{'total_tokens': num_tokens, 'ner_tokens': 0, 'o_tokens': 0}}
    tags_list = []
    logging.info(f"Writing NER CSV to {{output_csv_ner}} and tags to {{output_ner_tags}}...")
    try:
        with open(output_csv_ner,'w',encoding='utf-8',newline='') as fner, \\
             open(output_ner_tags,'w',encoding='utf-8') as ftags:
            wn = csv.writer(fner, quoting=csv.QUOTE_ALL)
            wn.writerow(['ID','TOKEN','NER'])

            for i, t in enumerate(doc):
                tid = i + 1; ttxt = str(t.text)
                nt = t.ent_type_ if t.ent_type_ else 'O'
                if nt != 'O': ner_stats['ner_tokens'] += 1
                else: ner_stats['o_tokens'] += 1
                wn.writerow([tid, ttxt, nt])
                tags_list.append(nt)

            ftags.write('\\n'.join(tags_list))
        logging.info(f"Finished writing NER CSV and tags file. NER tokens: {{ner_stats['ner_tokens']}}, O tokens: {{ner_stats['o_tokens']}}.")
    except Exception as csv_e:
        logging.error(f"Failed during NER CSV/Tags writing: {{csv_e}}", exc_info=True)
        sys.exit(1)

    logging.info(f"Writing NER stats summary to {{ner_summary_file}}...")
    try:
        ner_percentage = (ner_stats['ner_tokens'] / num_tokens * 100) if num_tokens > 0 else 0
        summary_data = {{
            "doc_id_original": old_id_str, "doc_id_new": new_id_str,
            "docbin_path": docbin_path, "ner_model_used": ner_model_name,
            "total_tokens": ner_stats['total_tokens'],
            "tokens_with_ner": ner_stats['ner_tokens'],
            "tokens_without_ner": ner_stats['o_tokens'],
            "ner_percentage": round(ner_percentage, 2)
        }}
        with open(ner_summary_file,'w',encoding='utf-8') as fsum:
            json.dump(summary_data, fsum, indent=2)
        logging.info(f"Finished writing NER stats summary.")
    except Exception as e:
        logging.warning(f"Failed write NER summary file: {{e}}", exc_info=True)

except Exception as e:
    logging.error(f"Unhandled exception in NER temporary script: {{e}}", exc_info=True)
    sys.exit(1)

logging.info(f"NER extraction script finished successfully for Doc ID: {{old_id_str}} -> {{new_id_str}}.")
sys.exit(0)
"""
    run_log_ctx = log_ctx.copy()
    run_log_ctx.update({
        'source_file': docbin_path,
        'destination_file': f"{os.path.basename(output_csv_ner)} & {os.path.basename(output_ner_tags)}",
        'file_type': 'ner-model-outputs'
    })
    return run_python_script_in_conda_env(conda_env, script_content, run_log_ctx, logger, timeout=600)


# --- Function to summarize mismatches (Added from your suggestion) ---
@timed_operation("summarize_mismatches")
def summarize_token_mismatches(output_base_dir: str, logger: Optional[FileOperationLogger] = None):
    """Collects and summarizes token mismatch information across the corpus."""
    base_path = Path(output_base_dir)
    info_files = list(base_path.glob("**/*_ner_mismatch_info.json"))
    error_files = list(base_path.glob("**/*_ner_error_info.json"))
    all_files = info_files + error_files
    script_logger.info(f"Found {len(info_files)} mismatch info files and {len(error_files)} error info files to summarize in {output_base_dir}.")

    if not all_files:
        script_logger.info("No mismatch or error info files found. Skipping summary.")
        return None

    total_docs_checked = 0; mismatched_docs_count = 0; error_docs_count = 0
    alignment_attempts = 0; alignment_successes = 0
    token_diff_stats = Counter(); alignment_statuses = Counter(); error_messages = Counter()
    processed_doc_ids = set()

    for info_file in all_files:
        try:
            with open(info_file, 'r', encoding='utf-8') as f: data = json.load(f)
            doc_id = data.get('document_id', info_file.stem.split('_')[0])
            if doc_id in processed_doc_ids: continue
            processed_doc_ids.add(doc_id)
            total_docs_checked += 1

            if info_file.name.endswith("_ner_error_info.json"):
                error_docs_count += 1
                error_messages[data.get('error_message', 'Unknown Error')] += 1
                continue

            if data.get('mismatch_detected', False): mismatched_docs_count += 1
            alignment_info = data.get('alignment_info')
            if alignment_info:
                 alignment_attempts += 1
                 status = alignment_info.get('status')
                 if status: alignment_statuses[status] += 1
                 if status in ('partial', 'success'): alignment_successes += 1

            main_tokens = data.get('main_model_tokens')
            ner_tags = data.get('ner_model_tags')
            if main_tokens is not None and ner_tags is not None:
                 token_diff_stats[ner_tags - main_tokens] += 1
            elif data.get('mismatch_detected', False):
                 script_logger.debug(f"Mismatch detected for {doc_id} but token counts missing in info file.")

        except json.JSONDecodeError as e: script_logger.warning(f"Error decoding JSON from file {info_file}: {e}")
        except Exception as e: script_logger.warning(f"Error processing info file {info_file}: {e}", exc_info=True)

    mismatch_percentage = (mismatched_docs_count / total_docs_checked * 100) if total_docs_checked else 0
    alignment_success_rate = (alignment_successes / alignment_attempts * 100) if alignment_attempts else 0

    summary = {
        "total_documents_with_info_file": total_docs_checked,
        "documents_with_ner_processing_error": error_docs_count,
        "documents_with_token_mismatch": mismatched_docs_count,
        "mismatch_percentage_of_checked": round(mismatch_percentage, 2),
        "alignment_attempts_on_mismatched": alignment_attempts,
        "alignment_successes_incl_partial": alignment_successes,
        "alignment_success_rate_incl_partial": round(alignment_success_rate, 2),
        "alignment_status_counts": dict(alignment_statuses),
        "token_difference_distribution": {str(k): v for k, v in sorted(token_diff_stats.items())},
        "ner_processing_error_counts": dict(error_messages)
    }

    summary_file = base_path / "corpus_token_mismatch_summary.json"
    log_ctx_summary = {'old_id': 'corpus', 'new_id': 'summary', 'corpus_prefix': 'all', 'operation_type': 'mismatch_summary', 'file_type': 'json', 'source_file': 'multiple', 'destination_file': str(summary_file)}

    try:
        with open(summary_file, 'w', encoding='utf-8') as f: json.dump(summary, f, indent=2)
        script_logger.info(f"Token mismatch summary written to: {summary_file}")
        summary_details = f"Checked:{total_docs_checked}, Errors:{error_docs_count}, Mismatched:{mismatched_docs_count} ({mismatch_percentage:.1f}%), Align Success Rate:{alignment_success_rate:.1f}% ({alignment_successes}/{alignment_attempts})"
        script_logger.info(summary_details)
        if logger: logger.log_operation(**log_ctx_summary, status='success', details=summary_details)
    except Exception as e:
        script_logger.error(f"Failed to write token mismatch summary: {e}", exc_info=True)
        if logger: logger.log_operation(**log_ctx_summary, status='failed', details=f"Error: {e}")
    return summary


# --- Helper Functions ---
def validate_input_file(file_path: str, file_type: str) -> bool:
    """Checks if an input file exists and is readable."""
    if not file_path: script_logger.error(f"Input {file_type} path is empty."); return False # Check for empty path
    abs_path = os.path.abspath(file_path) # Check absolute path
    if not os.path.exists(abs_path): script_logger.error(f"Input {file_type} file not found: {abs_path}"); return False
    if not os.path.isfile(abs_path): script_logger.error(f"Input {file_type} path is not a file: {abs_path}"); return False
    if not os.access(abs_path, os.R_OK): script_logger.error(f"Input {file_type} file not readable: {abs_path}"); return False
    script_logger.debug(f"Validated input file: {abs_path}")
    return True

def validate_output_dir(dir_path: str) -> bool:
    """Checks if output directory exists or can be created, and is writable."""
    if not dir_path: script_logger.error("Output directory path cannot be empty."); return False
    abs_path = os.path.abspath(dir_path) # Work with absolute path
    try:
        # Attempt to create the directory structure if it doesn't exist
        os.makedirs(abs_path, exist_ok=True)
        # Attempt to create and delete a temporary file to check write permissions
        test_file = os.path.join(abs_path, f".perm_check_{os.getpid()}") # Add PID for parallelism safety
        with open(test_file, "w") as f: f.write("test")
        os.remove(test_file)
        script_logger.debug(f"Validated output directory: {abs_path}")
        return True
    except OSError as e:
        script_logger.error(f"Cannot create or write to output dir '{abs_path}': {e}")
        return False
    except Exception as e: # Catch other potential errors
        script_logger.error(f"Unexpected error validating output dir '{abs_path}': {e}")
        return False


@timed_operation("load_index")
def load_index(index_path: str, log_context: Dict = None) -> Dict[str, str]:
    """Loads document_id -> processed_path index from CSV."""
    index = {}
    abs_index_path = os.path.abspath(index_path) # Use absolute path for clarity
    try:
        # Use low_memory=False if potential dtype issues or large files
        df = pd.read_csv(abs_index_path, low_memory=False)
        required_path_col = 'processed_path'
        id_col = 'document_id'

        if id_col not in df.columns:
             logging.error(f"Index file '{abs_index_path}' missing required column '{id_col}'.")
             return {}
        if required_path_col not in df.columns:
            logging.error(f"Index file '{abs_index_path}' missing required column '{required_path_col}'.")
            return {}

        # Iterate and populate the index dictionary
        duplicates = 0
        null_paths = 0
        for _, row in df.iterrows():
            doc_id = str(row[id_col]).strip() if pd.notna(row[id_col]) else None
            file_path = str(row[required_path_col]).strip() if pd.notna(row[required_path_col]) else None

            if not doc_id: continue # Skip if document_id is missing/empty

            if not file_path:
                logging.debug(f"Missing '{required_path_col}' for doc_id '{doc_id}' in '{abs_index_path}'.")
                null_paths += 1
                continue # Skip if path is missing

            abs_file_path = os.path.abspath(file_path) # Store absolute path

            if doc_id in index:
                logging.warning(f"Duplicate doc_id '{doc_id}' found in '{abs_index_path}'. Keeping first entry: '{index[doc_id]}'. Ignoring new path: '{abs_file_path}'")
                duplicates += 1
            else:
                index[doc_id] = abs_file_path

        if duplicates > 0: logging.warning(f"Found {duplicates} duplicate document IDs in {abs_index_path}.")
        if null_paths > 0: logging.warning(f"Found {null_paths} entries with missing paths in {abs_index_path}.")
        logging.info(f"Successfully loaded {len(index)} unique entries from index: {abs_index_path}")

    except FileNotFoundError:
        logging.error(f"Index file not found at '{abs_index_path}'")
        return {}
    except pd.errors.EmptyDataError:
        logging.error(f"Index file is empty: '{abs_index_path}'")
        return {}
    except Exception as e:
        logging.error(f"Error loading index file '{abs_index_path}': {type(e).__name__}: {e}", exc_info=True)
        return {}
    return index

@timed_operation("parse_mapping")
def parse_csv_mapping(csv_path: str, log_context: Dict = None) -> Dict[str, str]:
    """Parses document_id -> sort_id mapping from CSV."""
    mappings = {}
    abs_csv_path = os.path.abspath(csv_path)
    try:
        df = pd.read_csv(abs_csv_path, low_memory=False)
        id_col = 'document_id'
        sort_col = 'sort_id'

        if id_col not in df.columns:
            logging.error(f"Mapping file '{abs_csv_path}' missing required column '{id_col}'.")
            return {}
        if sort_col not in df.columns:
            logging.error(f"Mapping file '{abs_csv_path}' missing required column '{sort_col}'.")
            return {}

        duplicates = 0
        invalid_sort_ids = 0
        missing_sort_ids = 0
        for index, row in df.iterrows():
            old_id = str(row[id_col]).strip() if pd.notna(row[id_col]) else None
            new_id_val = row[sort_col] # Keep original type for checks

            if not old_id: continue # Skip if old_id is missing

            new_id = None
            if pd.notna(new_id_val):
                try:
                    # Ensure it's treated as int first, then string
                    new_id = str(int(float(new_id_val))) # Handle potential floats like '123.0'
                except (ValueError, TypeError):
                    logging.warning(f"Invalid non-numeric sort_id '{new_id_val}' for doc '{old_id}' in '{abs_csv_path}'. Skipping.")
                    invalid_sort_ids += 1
                    continue # Skip if sort_id is not a valid number
            else:
                # Log specifically if sort_id is NaN/None
                logging.warning(f"Missing sort_id (NaN/None) for doc '{old_id}' in '{abs_csv_path}'. Skipping.")
                missing_sort_ids += 1
                continue # Skip if sort_id is missing

            # If we have valid old_id and new_id
            if old_id in mappings:
                logging.warning(f"Duplicate document_id '{old_id}' in mapping '{abs_csv_path}'. Keeping first mapping: {old_id} -> {mappings[old_id]}. Ignoring new mapping to -> {new_id}")
                duplicates += 1
            else:
                mappings[old_id] = new_id

        if duplicates > 0: logging.warning(f"Found {duplicates} duplicate document IDs in {abs_csv_path}.")
        if invalid_sort_ids > 0: logging.warning(f"Found {invalid_sort_ids} invalid sort IDs in {abs_csv_path}.")
        if missing_sort_ids > 0: logging.warning(f"Found {missing_sort_ids} missing sort IDs in {abs_csv_path}.")
        logging.info(f"Successfully loaded {len(mappings)} unique mappings from: {abs_csv_path}")

    except FileNotFoundError:
        logging.error(f"Mapping file not found at '{abs_csv_path}'"); return {}
    except pd.errors.EmptyDataError:
        logging.error(f"Mapping file is empty: '{abs_csv_path}'")
        return {}
    except Exception as e:
        logging.error(f"Error parsing mapping file '{abs_csv_path}': {type(e).__name__}: {e}", exc_info=True)
        return {}
    return mappings


def find_original_text(doc_id: str, base_dir: str) -> Optional[str]:
    """Attempts to find the original .txt file based on doc_id and base_dir structure."""
    # Use Path for robustness
    base_path = Path(base_dir).resolve() # Resolve to absolute path
    source_sub_dir_name = "cleaned_parsed_data" # THIS SHOULD MATCH THE --base-dir ARGUMENT'S INTENTION
    # The --base-dir argument was 'dat/greek/cleaned_parsed_data/'
    # So, we should look *directly* in base_path and potential corpus subdirs within it
    # source_sub_dir = base_path / source_sub_dir_name # REMOVE THIS LINE

    corpus_prefix_match = re.match(r'^([a-zA-Z0-9\-_]+?)_', doc_id)
    corpus_prefix = corpus_prefix_match.group(1) if corpus_prefix_match else None

    possible_paths = []

    # Strategy 1: Look in corpus-specific subdirectory *within* base_dir
    if corpus_prefix:
        corpus_dir = base_path / corpus_prefix # Look in e.g., .../cleaned_parsed_data/perseus/
        filename_part = doc_id.replace(f"{corpus_prefix}_", "", 1)
        # Path using just the filename part after prefix
        possible_paths.append(corpus_dir / f"{filename_part}.txt")
        # Path using the full original doc_id within the corpus dir
        possible_paths.append(corpus_dir / f"{doc_id}.txt")

    # Strategy 2: Look directly in the base_dir itself (e.g., if no prefix or flat structure)
    # Path using the full original doc_id within the base dir
    possible_paths.append(base_path / f"{doc_id}.txt")


    # Check existence for each potential path
    for potential_path in possible_paths:
        script_logger.debug(f"Checking for original text at: {potential_path}")
        if potential_path.is_file():
            script_logger.info(f"Found original text for '{doc_id}' at: {potential_path}")
            return str(potential_path) # Return as string path

    script_logger.warning(f"Could not find original text file for doc_id '{doc_id}' using patterns in base_dir '{base_dir}'.")
    return None


# --- Helper to get expected output file paths ---
def get_expected_output_paths(output_base_dir: str, corpus_prefix: str, new_id: str) -> Dict[str, str]:
    """Returns a dictionary of expected *absolute* output file paths for a given new_id."""
    abs_output_base_dir = os.path.abspath(output_base_dir)
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
        "temp_ner_tags_file": os.path.join(annotations_dir, f".{new_id}_temp_ner_tags.txt"),
        "source_original_txt": os.path.join(texts_dir, f"{new_id}-original.txt"),
        "ner_mismatch_info": os.path.join(annotations_dir, f"{new_id}_ner_mismatch_info.json"),
        "ner_stats_file": os.path.join(annotations_dir, f"{new_id}_ner_stats.json"),
        "ner_error_info": os.path.join(annotations_dir, f"{new_id}_ner_error_info.json"),
    }

# --- START REPLACEMENT: process_document ---
# --- Modified Main Processing Logic ---
@timed_operation("process_document")
def process_document(
    old_id: str, new_id: str, base_dir: str, output_base_dir: str,
    main_index: Dict[str, str], ner_index: Dict[str, str],
    main_env: str, ner_env: str, logger: Optional[FileOperationLogger] = None,
    overwrite: bool = False, use_locking: bool = False,
    log_context: Dict = None
) -> Tuple[bool, str]:
    """Processes a single document based on pre-check results."""
    overall_success = True; final_status_msg = "Success"; temp_files_to_clean = []
    corpus_prefix_match = re.match(r'^([a-zA-Z0-9\-_]+?)_', old_id)
    corpus_prefix = corpus_prefix_match.group(1) if corpus_prefix_match else "unknown_corpus"

    # --- Safer log_context handling ---
    # Start with essential IDs and prefix
    current_log_context = {'old_id': old_id, 'new_id': new_id, 'corpus_prefix': corpus_prefix}
    # If a context was passed in, update ONLY with keys NOT already set or critical
    if log_context:
        for key, value in log_context.items():
            if key not in current_log_context: # Avoid overwriting core IDs/prefix
                current_log_context[key] = value
    # --- End safer log_context handling ---


    if logger: logger.log_operation(**current_log_context, operation_type="process_start", file_type="document", status="info", details=f"Processing (O={overwrite}, L={use_locking})")

    # --- Path Retrieval ---
    main_docbin_path = main_index.get(old_id)
    if not main_docbin_path or not os.path.isfile(main_docbin_path):
         if logger: logger.log_operation(**current_log_context, operation_type="lookup", file_type="main_docbin", status="failed", details="Main docbin path invalid/not found during processing step!")
         return False, "Failed: Main DocBin vanished post-check"

    ner_docbin_path = ner_index.get(old_id)
    ner_docbin_path_resolved = None
    if ner_docbin_path and os.path.isfile(ner_docbin_path): ner_docbin_path_resolved = ner_docbin_path
    elif ner_docbin_path:
        if logger: logger.log_operation(**current_log_context, operation_type="lookup", file_type="ner_docbin", status="warning", details="NER index path invalid/not found during processing step.")
    else: script_logger.debug(f"[{old_id}->{new_id}] No NER docbin path found in index.")

    source_txt_path = find_original_text(old_id, base_dir)
    paths = get_expected_output_paths(output_base_dir, corpus_prefix, new_id)
    texts_dir = paths["texts_dir"]; annotations_dir = paths["annotations_dir"]
    # Get specific paths needed later
    output_txt_joined=paths["output_txt_joined"]; output_txt_fullstop=paths["output_txt_fullstop"]
    output_csv_lemma=paths["output_csv_lemma"]; output_csv_upos=paths["output_csv_upos"]
    output_csv_stop=paths["output_csv_stop"]; output_csv_dot=paths["output_csv_dot"]
    output_csv_ner=paths["output_csv_ner"]; output_conllu=paths["output_conllu"]
    temp_ner_tags_file=Path(paths["temp_ner_tags_file"]); temp_files_to_clean.append(str(temp_ner_tags_file))
    ner_tags_lock_path=str(temp_ner_tags_file)+".lock"
    dest_original_txt = paths["source_original_txt"]
    # --- Get lock directory ---
    lock_dir = os.path.dirname(ner_tags_lock_path)

    # --- Directory Creation ---
    try:
        # Create annotations dir first, as lock file goes there
        os.makedirs(annotations_dir, exist_ok=True)
        script_logger.debug(f"[{old_id}->{new_id}] Ensured annotations directory exists: {annotations_dir}")
        # Then create texts dir
        os.makedirs(texts_dir, exist_ok=True)
        script_logger.debug(f"[{old_id}->{new_id}] Ensured texts directory exists: {texts_dir}")
    except OSError as e:
         # Use current_log_context for logging
         if logger: logger.log_operation(**current_log_context, operation_type="create_dir", file_type="directory", status="failed", details=f"Dir creation error for {texts_dir} or {annotations_dir}: {e}")
         return False, f"Failed: Dir creation error {e}"

    try:
        # Determine lock class *once* per document process
        use_real_lock = use_locking and FILELOCK_AVAILABLE
        LockClass = filelock.FileLock if use_real_lock else DummyFileLock
        LockTimeoutError = FileLockTimeout # Use the globally assigned Timeout (real or dummy)
        # --- ADD DEBUG LOG ---
        script_logger.debug(f"[{old_id}->{new_id}] Using LockClass: {LockClass.__name__} (use_locking={use_locking}, FILELOCK_AVAILABLE={FILELOCK_AVAILABLE})")
        # --- END DEBUG LOG ---

        ner_lock_instance = None # Define variable to hold lock instance if needed

        # --- NER Extraction ---
        ner_tags_available = False
        if ner_docbin_path_resolved:
            script_logger.info(f"[{old_id}->{new_id}] Attempting NER extraction & lock...")

            # --- ADD PRE-LOCK CHECKS ---
            try:
                script_logger.debug(f"[{old_id}->{new_id}] Checking lock directory: {lock_dir}")
                if not os.path.exists(lock_dir):
                    script_logger.error(f"[{old_id}->{new_id}] Lock directory does not exist: {lock_dir}")
                    raise OSError(f"Lock directory missing: {lock_dir}") # Raise specific error
                if not os.access(lock_dir, os.W_OK):
                    script_logger.error(f"[{old_id}->{new_id}] No write access to lock directory: {lock_dir}")
                    raise PermissionError(f"No write access to lock directory: {lock_dir}") # Raise specific error
                script_logger.debug(f"[{old_id}->{new_id}] Lock directory check OK.")

                script_logger.debug(f"[{old_id}->{new_id}] Instantiating lock: {LockClass.__name__} for {ner_tags_lock_path}")
                ner_lock_instance = LockClass(ner_tags_lock_path, timeout=10) # Assign instance here
            except Exception as pre_lock_e:
                # Catch errors during pre-checks or lock instantiation
                script_logger.error(f"!!! Pre-Lock/Instantiation Exception ({type(pre_lock_e).__name__}) for doc {old_id}->{new_id}: {pre_lock_e}", exc_info=True)
                if logger: logger.log_operation(
                    old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                    operation_type="lock_setup", file_type="ner-tags-write", status="failed",
                    details=f"Error setting up lock for {os.path.basename(ner_tags_lock_path)}: {pre_lock_e} (Type: {type(pre_lock_e).__name__})"
                )
                overall_success = False; final_status_msg = "Failed: Lock setup error"
                # Skip to finally block if setup fails
                ner_docbin_path_resolved = None # Use this flag to skip the acquire block
            # --- END PRE-LOCK CHECKS ---

            # Only attempt acquire if pre-checks passed and lock was instantiated
            if overall_success and ner_docbin_path_resolved and ner_lock_instance: # Check flag and instance
                try:
                    script_logger.debug(f"[{old_id}->{new_id}] Acquiring NER write lock: {ner_tags_lock_path}")
                    with ner_lock_instance.acquire(): # Use the instantiated lock
                        script_logger.debug(f"[{old_id}->{new_id}] NER write lock acquired. Running extraction...")
                        # Pass current_log_context here
                        if extract_ner_model_outputs(ner_env, ner_docbin_path_resolved, output_csv_ner, str(temp_ner_tags_file), logger, **current_log_context):
                            ner_tags_available = True
                        else:
                            overall_success = False; final_status_msg = "Failed: NER extraction"
                    script_logger.debug(f"[{old_id}->{new_id}] NER write lock released.")
                except (LockTimeoutError, Exception) as lock_e:
                    script_logger.error(f"!!! Lock Exception caught during NER Write Acquire ({type(lock_e).__name__}) for doc {old_id}->{new_id}: {lock_e}", exc_info=True) # Log traceback
                    lock_fail_details = f"Timeout acquiring lock {os.path.basename(ner_tags_lock_path)}" if isinstance(lock_e, LockTimeoutError) else f"Error acquiring lock {os.path.basename(ner_tags_lock_path)}: {lock_e}"
                    # Use explicit logging call (FIX from previous step)
                    if logger: logger.log_operation(
                        old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                        operation_type="lock", file_type="ner-tags-write", status="failed",
                        details=f"{lock_fail_details} (Exception Type: {type(lock_e).__name__})"
                    )
                    overall_success = False; final_status_msg = f"Failed: NER lock {'timeout' if isinstance(lock_e, LockTimeoutError) else 'error'}"
        else:
             if logger: logger.log_operation(**current_log_context, operation_type="extract", file_type="ner-model-outputs", status="skipped", details="NER docbin not resolved or found during processing step.")

        # --- Main Model Extraction ---
        if overall_success:
            ner_tags_input_path = None
            script_logger.debug(f"[{old_id}->{new_id}] Attempting main extraction & read lock...")
            ner_read_lock = LockClass(ner_tags_lock_path, timeout=10) # Use determined LockClass again
            try:
                script_logger.debug(f"[{old_id}->{new_id}] Acquiring NER read lock: {ner_tags_lock_path}")
                with ner_read_lock.acquire():
                    script_logger.debug(f"[{old_id}->{new_id}] NER read lock acquired.")
                    if temp_ner_tags_file.is_file(): ner_tags_input_path = str(temp_ner_tags_file)
                    elif ner_tags_available:
                         if logger: logger.log_operation(**current_log_context, operation_type="read", file_type="temp_ner_tags", status="warning", details="NER extraction reported success but temp tags file not found before main extraction.")
                script_logger.debug(f"[{old_id}->{new_id}] NER read lock released (implicitly by exiting 'with').")

                # Extraction happens *after* releasing the read lock
                script_logger.info(f"[{old_id}->{new_id}] Extracting Main Model outputs (using NER tags: {ner_tags_input_path is not None})...")
                # Pass current_log_context here
                if not extract_main_model_outputs(
                    main_env, main_docbin_path,
                    output_txt_joined, output_txt_fullstop,
                    output_csv_lemma, output_csv_upos, output_csv_stop, output_csv_dot,
                    output_conllu, ner_tags_input_path, new_id, logger, **current_log_context
                ):
                    overall_success = False; final_status_msg = "Failed: Main model extraction"

            except (LockTimeoutError, Exception) as lock_e:
                script_logger.error(f"!!! Lock Exception caught during NER Read Acquire ({type(lock_e).__name__}) for doc {old_id}->{new_id}: {lock_e}", exc_info=True) # Log traceback
                lock_fail_details = f"Timeout acquiring lock {os.path.basename(ner_tags_lock_path)} for read" if isinstance(lock_e, LockTimeoutError) else f"Error acquiring lock {os.path.basename(ner_tags_lock_path)} for read: {lock_e}"
                # Use explicit logging call (FIX from previous step)
                if logger: logger.log_operation(
                    old_id=old_id, new_id=new_id, corpus_prefix=corpus_prefix,
                    operation_type="lock", file_type="conllu-ner-read", status="failed",
                    details=f"{lock_fail_details} (Exception Type: {type(lock_e).__name__})"
                )
                overall_success = False; final_status_msg = f"Failed: CoNLLU lock {'timeout' if isinstance(lock_e, LockTimeoutError) else 'error'}"

        # --- Copy Original Text ---
        # Use current_log_context for these logs too
        if source_txt_path:
            if overwrite or not os.path.exists(dest_original_txt):
                try:
                    shutil.copy2(source_txt_path, dest_original_txt)
                    if logger: logger.log_operation(**current_log_context, operation_type="copy", file_type="source_txt", source_file=source_txt_path, destination_file=dest_original_txt, status="success")
                except Exception as e:
                    if logger: logger.log_operation(**current_log_context, operation_type="copy", file_type="source_txt", source_file=source_txt_path, destination_file=dest_original_txt, status="failed", details=str(e))
            else:
                 if logger: logger.log_operation(**current_log_context, operation_type="copy", file_type="source_txt", status="skipped", details="Destination original text exists (overwrite=False)")
        else:
            if logger: logger.log_operation(**current_log_context, operation_type="copy", file_type="source_txt", status="skipped", details="Source original text file not found.")

    finally:
        # Use current_log_context for cleanup logs
        script_logger.debug(f"[{old_id}->{new_id}] Cleaning up temporary files...")
        for temp_file_path_str in temp_files_to_clean:
            temp_file_p = Path(temp_file_path_str)
            if temp_file_p.exists():
                try: temp_file_p.unlink(); script_logger.debug(f"[{old_id}->{new_id}] Removed temp file: {temp_file_p.name}")
                except OSError as e:
                    if logger: logger.log_operation(**current_log_context, operation_type="cleanup", file_type="temp_file", status="failed", details=f"Error removing {temp_file_p.name}: {e}")
        lock_file_p = Path(ner_tags_lock_path)
        # Check if the *real* lock file exists before trying to remove it
        if use_locking and FILELOCK_AVAILABLE and lock_file_p.exists():
            try: lock_file_p.unlink(); script_logger.debug(f"[{old_id}->{new_id}] Removed lock file: {lock_file_p.name}")
            except OSError as e:
                 if logger: logger.log_operation(**current_log_context, operation_type="cleanup", file_type="lock_file", status="failed", details=f"Failed remove lock {lock_file_p.name}: {e}")

    # Use current_log_context for the final log
    if logger: logger.log_operation(**current_log_context, operation_type="process_end", file_type="document", status="success" if overall_success else "failed", details=f"Finished. Final Status: {final_status_msg}")
    return overall_success, final_status_msg
# --- END REPLACEMENT: process_document ---


# --- (#8) Graceful Shutdown Handler ---
_shutdown_requested = False; _executor_ref = None
def setup_graceful_shutdown():
    global _executor_ref
    def handler(sig, frame):
        global _shutdown_requested, _executor_ref
        if not _shutdown_requested:
            signal_name = signal.Signals(sig).name
            msg = f"Shutdown requested ({signal_name}). Waiting for active tasks...";
            logging.warning(msg); script_logger.warning(msg); print(f"\n{msg}", file=sys.stderr)
            _shutdown_requested = True
            if _executor_ref:
                script_logger.info("Requesting executor shutdown (wait=True)...")
                try:
                    _executor_ref.shutdown(wait=True, cancel_futures=False)
                    script_logger.info("Executor shutdown complete.")
                except Exception as e: script_logger.error(f"Error during executor shutdown: {e}")
            script_logger.warning("Exiting due to shutdown signal.")
            sys.exit(0)
        else: script_logger.info("Shutdown already in progress.")
    signal.signal(signal.SIGINT, handler); signal.signal(signal.SIGTERM, handler)


# --- Pre-check Function ---
@timed_operation("determine_tasks")
def determine_tasks_to_run(
    mappings: Dict[str, str],
    main_index: Dict[str, str],
    ner_index: Dict[str, str], # Keep NER index for context, e.g., logging path issues
    output_base_dir: str,
    overwrite: bool,
    logger: Optional[FileOperationLogger] = None,
    log_context: Dict = None # Pass parent context if needed
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Pre-checks mappings against indices and existing files."""
    tasks_to_run = []
    skipped_missing_index_count = 0; skipped_already_complete_count = 0
    total_potential_tasks = len(mappings)
    script_logger.info(f"Pre-checking {total_potential_tasks} potential documents based on mapping...")
    check_iterator = tqdm.tqdm(mappings.items(), desc="Pre-checking tasks", total=total_potential_tasks, unit="doc", dynamic_ncols=True, disable=total_potential_tasks <= 50)

    for old_id, new_id in check_iterator:
        corpus_prefix_match = re.match(r'^([a-zA-Z0-9\-_]+?)_', old_id)
        corpus_prefix = corpus_prefix_match.group(1) if corpus_prefix_match else "unknown_corpus"
        current_log_ctx = {'old_id': old_id, 'new_id': new_id, 'corpus_prefix': corpus_prefix}

        main_docbin_path_check = main_index.get(old_id)
        if not main_docbin_path_check:
             if logger: logger.log_operation(**current_log_ctx, operation_type="pre_check", file_type="main_docbin", status="skipped", details="Doc ID not found in main index.")
             skipped_missing_index_count += 1; continue
        if not os.path.isfile(main_docbin_path_check):
             if logger: logger.log_operation(**current_log_ctx, operation_type="pre_check", file_type="main_docbin", source_file=main_docbin_path_check, status="skipped", details="Main docbin file path from index is invalid or file not found.")
             skipped_missing_index_count += 1; continue

        ner_docbin_path_check = ner_index.get(old_id)
        if ner_docbin_path_check and not os.path.isfile(ner_docbin_path_check):
             if logger: logger.log_operation(**current_log_ctx, operation_type="pre_check", file_type="ner_docbin", source_file=ner_docbin_path_check, status="warning", details="NER docbin file path from index is invalid or file not found. Processing will continue without NER.")

        if overwrite:
            if logger: logger.log_operation(**current_log_ctx, operation_type="pre_check", file_type="document", status="queued", details="Added to queue (overwrite=True).")
            tasks_to_run.append({'old_id': old_id, 'new_id': new_id}); continue

        paths = get_expected_output_paths(output_base_dir, corpus_prefix, new_id)
        key_output_files = [paths["output_conllu"]]
        all_key_files_exist = True; missing_file_details = ""
        for f_path in key_output_files:
            if not os.path.exists(f_path):
                all_key_files_exist = False
                missing_file_details = f"Missing key output file: {os.path.basename(f_path)}"
                script_logger.debug(f"[{old_id}->{new_id}] {missing_file_details}")
                break

        if all_key_files_exist:
            if logger: logger.log_operation(**current_log_ctx, operation_type="pre_check", file_type="document_output", status="skipped", details="All key output files exist (overwrite=False).")
            skipped_already_complete_count += 1
        else:
            if logger: logger.log_operation(**current_log_ctx, operation_type="pre_check", file_type="document_output", status="queued", details=f"Added to queue ({missing_file_details}).")
            tasks_to_run.append({'old_id': old_id, 'new_id': new_id})

    script_logger.info(f"Pre-check complete. Determined {len(tasks_to_run)} tasks need processing.")
    if skipped_missing_index_count > 0: script_logger.info(f"Skipped {skipped_missing_index_count} tasks due to missing/invalid main docbin index entry or file.")
    if skipped_already_complete_count > 0: script_logger.info(f"Skipped {skipped_already_complete_count} tasks because key output files already exist (overwrite=False).")
    return tasks_to_run, skipped_missing_index_count, skipped_already_complete_count


# --- Main Execution ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Reorganize corpus documents...", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mapping-csv", required=True, help="Path to CSV mapping 'document_id' to 'sort_id'.")
    parser.add_argument("--main-index-csv", required=True, help="Path to CSV index for main model DocBins ('document_id', 'processed_path').")
    parser.add_argument("--ner-index-csv", required=True, help="Path to CSV index for NER model DocBins ('document_id', 'processed_path').")
    parser.add_argument("--base-dir", required=True, help="Path to base directory containing original source data (e.g., 'dat/greek/cleaned_parsed_data/').")
    parser.add_argument("--output-dir", required=True, help="Path to base directory for reorganized output.")
    parser.add_argument("--main-env", required=True, help="Name of Conda environment for the main spaCy model (e.g., 'grc_proiel_trf').")
    parser.add_argument("--ner-env", required=True, help="Name of Conda environment for the NER spaCy model (e.g., 'grc_ner_trf').")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files. If False, skip documents where key output files exist.")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 1, help="Number of parallel processes. Defaults to number of CPU cores.")
    parser.add_argument("--use-locking", action="store_true", help=f"Use file locking for parallel safety on temp files (Requires 'filelock': {'Available' if FILELOCK_AVAILABLE else 'NOT FOUND - Locking Disabled'}).")
    parser.add_argument("--log-file", default="reorganization_log.csv", help="Path for the detailed CSV log file.")
    parser.add_argument("--wandb-project", default="corpus-reorganization", help="WandB project name for logging.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    args = parser.parse_args()

    # --- Input Validation ---
    script_logger.info("Validating inputs...")
    valid_inputs = True
    if not validate_input_file(args.mapping_csv, "Mapping CSV"): valid_inputs = False
    if not validate_input_file(args.main_index_csv, "Main Index CSV"): valid_inputs = False
    if not validate_input_file(args.ner_index_csv, "NER Index CSV"): valid_inputs = False
    if not os.path.isdir(args.base_dir): logging.warning(f"Base directory '{args.base_dir}' does not exist or is not a directory.")
    if not validate_output_dir(args.output_dir): valid_inputs = False
    if args.num_workers <= 0: logging.warning(f"Invalid --num-workers value ({args.num_workers}). Setting to 1."); args.num_workers = 1
    if args.use_locking and not FILELOCK_AVAILABLE: logging.warning("--use-locking specified but 'filelock' is not installed. Disabling file locking."); args.use_locking = False
    if not valid_inputs: script_logger.error("Input validation failed. Exiting."); sys.exit(1)
    script_logger.info("Input validation successful.")

    # --- Initialize Logger and Timers ---
    start_time = time.time()
    logger = FileOperationLogger(log_file_path=args.log_file, use_wandb=(not args.no_wandb), wandb_project=args.wandb_project)

    # --- Log Script Arguments ---
    script_logger.info("=" * 60); script_logger.info(" Corpus Reorganization Script Started"); script_logger.info("=" * 60)
    script_logger.info(f" Run Name: {logger.run_name}")
    script_logger.info(f" Overwrite Mode: {args.overwrite}")
    script_logger.info(f" Parallel Workers: {args.num_workers}")
    script_logger.info(f" File Locking: {'Enabled' if args.use_locking else 'Disabled'}")
    script_logger.info(f" Mapping CSV: {os.path.abspath(args.mapping_csv)}")
    script_logger.info(f" Main Index CSV: {os.path.abspath(args.main_index_csv)}")
    script_logger.info(f" NER Index CSV: {os.path.abspath(args.ner_index_csv)}")
    script_logger.info(f" Base Source Dir: {os.path.abspath(args.base_dir)}")
    script_logger.info(f" Output Base Dir: {os.path.abspath(args.output_dir)}")
    script_logger.info(f" Main Conda Env: {args.main_env}")
    script_logger.info(f" NER Conda Env: {args.ner_env}")
    script_logger.info(f" Log File: {os.path.abspath(args.log_file) if logger.log_file_path else 'Disabled'}")
    script_logger.info(f" WandB Logging: {'Enabled' if logger.use_wandb else 'Disabled'}")
    if logger.use_wandb and logger.wandb_run_url: script_logger.info(f" WandB Project: {args.wandb_project} (Run URL: {logger.wandb_run_url})")
    script_logger.info("-" * 60)

    # --- Warn if indices are same ---
    if os.path.abspath(args.main_index_csv) == os.path.abspath(args.ner_index_csv): logging.warning("="*60); logging.warning("Warning: Main index and NER index files appear to be the same."); logging.warning("="*60)

    # --- Update WandB Config ---
    if logger.use_wandb and wandb.run:
         try:
              args_for_config=vars(args).copy()
              current_wandb_log_file_actual=wandb.config.get("log_file_actual")
              if "log_file" in args_for_config and current_wandb_log_file_actual is not None:
                  del args_for_config["log_file"]
              wandb.config.update(args_for_config, allow_val_change=True)
              script_logger.info("Updated WandB config with script arguments.")
         except Exception as e: logging.warning(f"Failed to fully update wandb config: {e}")

    # --- Load Mappings and Indices ---
    script_logger.info("Loading mappings and indices...")
    base_log_ctx = {'operation_type': 'setup'}
    mappings = parse_csv_mapping(args.mapping_csv, log_context=base_log_ctx)
    main_index = load_index(args.main_index_csv, log_context=base_log_ctx)
    ner_index = load_index(args.ner_index_csv, log_context=base_log_ctx)
    script_logger.info(f"Loaded {len(mappings)} unique mappings.")
    script_logger.info(f"Loaded {len(main_index)} unique main index entries.")
    script_logger.info(f"Loaded {len(ner_index)} unique NER index entries.")

    # --- Exit Condition Checks ---
    if not mappings:
        script_logger.error("Mapping file loaded 0 entries. Cannot proceed. Exiting.")
        if logger: logger.summarize_and_close()
        sys.exit(1)
    if not main_index:
        script_logger.error("Main index file loaded 0 entries. Cannot proceed. Exiting.")
        if logger: logger.summarize_and_close()
        sys.exit(1)
    if not ner_index: script_logger.warning("NER index file loaded 0 entries. Processing will continue without NER data integration.")

    # --- Determine Tasks to Run (Pre-check) ---
    tasks_to_process_info, skipped_missing_index, skipped_already_complete = determine_tasks_to_run(
        mappings=mappings, main_index=main_index, ner_index=ner_index,
        output_base_dir=args.output_dir, overwrite=args.overwrite, logger=logger, log_context=base_log_ctx
    )

    # --- Construct Full Task Arguments ---
    script_logger.info("Constructing full task arguments for processing...")
    full_tasks = []
    for task_info in tasks_to_process_info:
        old_id = task_info['old_id']; new_id = task_info['new_id']
        corpus_prefix = (re.match(r'^([a-zA-Z0-9\-_]+?)_', old_id).group(1) if re.match(r'^([a-zA-Z0-9\-_]+?)_', old_id) else "unknown_corpus")
        full_tasks.append({
            'old_id': old_id, 'new_id': new_id, 'base_dir': args.base_dir,
            'output_base_dir': args.output_dir, 'main_index': main_index, 'ner_index': ner_index,
            'main_env': args.main_env, 'ner_env': args.ner_env, 'logger': logger,
            'overwrite': args.overwrite, 'use_locking': args.use_locking,
            'log_context': {'old_id': old_id, 'new_id': new_id, 'corpus_prefix': corpus_prefix}
        })
    script_logger.info(f"Prepared {len(full_tasks)} full task argument sets.")

    # --- Setup Worker Wrapper and Shutdown Handler ---
    def process_document_wrapper(task_args_dict):
        try: return process_document(**task_args_dict)
        except Exception as worker_exc:
            worker_old_id = task_args_dict.get('old_id', 'unknown'); worker_new_id = task_args_dict.get('new_id', 'unknown')
            script_logger.error(f"!!! Uncaught exception in worker for {worker_old_id}->{worker_new_id}: {type(worker_exc).__name__}: {worker_exc}")
            script_logger.error(traceback.format_exc())
            log_ctx_err = task_args_dict.get('log_context', {'old_id': worker_old_id, 'new_id': worker_new_id})
            if logger: logger.log_operation(**log_ctx_err, operation_type="process_uncaught", file_type="document", status="failed", details=f"Worker Exception: {type(worker_exc).__name__}: {worker_exc}")
            return False, f"Failed: Worker Exception {type(worker_exc).__name__}"
    setup_graceful_shutdown()

    # --- Execute Processing ---
    processed_count = len(full_tasks); failed_count = 0
    script_logger.info(f"--- Starting document processing for {processed_count} tasks ---")
    executor = None; pbar = None
    try:
        if processed_count == 0: script_logger.info("No documents require processing...")
        elif args.num_workers > 1:
             script_logger.info(f"Starting parallel processing with {args.num_workers} workers.")
             pbar = tqdm.tqdm(total=processed_count, desc="Processing Docs", unit="doc", dynamic_ncols=True)
             with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                 _executor_ref = executor
                 futures = {executor.submit(process_document_wrapper, task): task for task in full_tasks}
                 for future in as_completed(futures):
                     task_info = futures[future]; task_id_str = f"{task_info.get('old_id','?')}->{task_info.get('new_id','?')}"
                     if _shutdown_requested: script_logger.warning(f"Shutdown requested, stopping processing of results for task {task_id_str}"); break
                     try:
                         success, final_status_msg = future.result()
                         if not success: failed_count += 1
                         pbar.set_postfix_str(f"Last: {task_id_str} ({'OK' if success else 'FAIL'})", refresh=True)
                     except Exception as exc:
                          script_logger.error(f'Task {task_id_str} generated exception during future.result(): {exc}')
                          failed_count += 1
                          pbar.set_postfix_str(f"Last: {task_id_str} (EXCEPTION)", refresh=True)
                     finally:
                          if pbar: pbar.update(1)
             _executor_ref = None
        else: # Sequential processing
            script_logger.info("Starting sequential processing (1 worker).")
            pbar = tqdm.tqdm(full_tasks, desc="Processing Docs", unit="doc", dynamic_ncols=True)
            for task in pbar:
                if _shutdown_requested: script_logger.warning("Shutdown requested during sequential processing, stopping."); break
                task_id_str = f"{task.get('old_id','?')}->{task.get('new_id','?')}"
                pbar.set_postfix_str(f"Current: {task_id_str}", refresh=False)
                success, final_status_msg = process_document_wrapper(task)
                if not success: failed_count += 1
                pbar.set_postfix_str(f"Last: {task_id_str} ({'OK' if success else 'FAIL'})", refresh=True)

    except KeyboardInterrupt: script_logger.warning("\nKeyboardInterrupt received. Initiating shutdown...")
    finally:
        if pbar: pbar.close(); script_logger.info("Progress bar closed.")
        _executor_ref = None

    # --- Final Steps ---
    script_logger.info("--- Document processing loop finished ---")
    if processed_count > 0: script_logger.info("--- Generating Token Mismatch Summary ---"); summarize_token_mismatches(args.output_dir, logger)
    else: script_logger.info("Skipping mismatch summary as no documents were processed.")

    # --- Final Summary Logging ---
    end_time = time.time(); duration = end_time - start_time
    attempted_processing = processed_count # Clarity: number of tasks submitted

    # Correct construction of the multi-line f-string summary
    summary_msg = (
        "\n" + "=" * 60 +
        "\n--- Reorganization Summary ---" +
        f"\nTotal documents in mapping file: {len(mappings)}" +
        f"\nDocuments skipped (missing/invalid main index/docbin): {skipped_missing_index}" +
        f"\nDocuments skipped (outputs exist, overwrite=False): {skipped_already_complete}" +
        f"\n----------------------------------" +
        f"\nDocuments submitted for processing: {attempted_processing}" +
        # Note: No trailing '+' or '#' on these lines within the parentheses
        f"\n  -> Successfully processed: {attempted_processing - failed_count}" +
        f"\n  -> Failed processing: {failed_count}" +
        f"\n----------------------------------" +
        f"\nTotal execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)" +
        "\n" + "=" * 60
    ) # The parentheses close the multi-line string construction

    script_logger.info(summary_msg); print(summary_msg) # Print to console as well

    # Finalize logging (writes summary, uploads artifacts, closes wandb)
    summary_stats = logger.summarize_and_close()

    # Provide final pointers to logs
    if logger.log_file_path and os.path.exists(logger.log_file_path):
        script_logger.info(f"Detailed CSV log saved to: {logger.log_file_path}")
    elif args.log_file:
         script_logger.warning(f"CSV log file was intended for {args.log_file} but may not have been created due to errors.")

    if not args.no_wandb:
        # Check if wandb run URL exists in the final summary (may not if init failed)
        wandb_url = summary_stats.get('wandb_run_url', None) # Assuming logger adds this
        if wandb_url:
            script_logger.info(f"W&B Run URL: {wandb_url}")
        elif logger.use_wandb: # Check if it was intended but failed
            script_logger.info("W&B logging was enabled but run may have ended or failed to initialize.")
        else:
            script_logger.info("W&B logging was disabled.")

    script_logger.info("Script finished."); print("\nScript finished.")
    # Optional: Exit with non-zero code if there were failures
    # sys.exit(1 if failed_count > 0 else 0)
