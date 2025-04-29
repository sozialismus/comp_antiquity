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

# Optional dependency for file locking
try:
    import filelock
    FILELOCK_AVAILABLE = True
except ImportError:
    FILELOCK_AVAILABLE = False
    logging.warning("Optional dependency 'filelock' not found. File locking disabled.")
    class DummyFileLock:
        def __init__(self, lock_file, timeout=-1): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_value, traceback): pass
        def acquire(self, timeout=None, poll_interval=0.05): return self
        def release(self, force=False): pass
    filelock = type('module', (), {'FileLock': DummyFileLock, 'Timeout': type('Timeout', (Exception,), {})})()


import pandas as pd
import tqdm
import wandb

# --- Configure standard logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
script_logger = logging.getLogger("ReorgScript")


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
                if args and isinstance(args[-1], dict): log_ctx = args[-1]
                if not log_ctx and kwargs.get('log_context'): log_ctx = kwargs['log_context']
                doc_info = f" (Doc: {log_ctx.get('old_id','?')}->{log_ctx.get('new_id','?')})" if log_ctx else ""
                func_logger.info(f"Finished in {duration:.3f}s{doc_info}")
                if wandb.run: wandb.log({f"timing/{operation_name}_duration_sec": duration}, commit=False)
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
    log_file_type = log_context.get('file_type', 'unknown_script')
    try:
        with open(temp_script_path, 'w', encoding='utf-8') as f_script:
            f_script.write("#!/usr/bin/env python3\n"); f_script.write("# -*- coding: utf-8 -*-\n"); f_script.write(script_content)
        cmd = f"conda run -n {conda_env} python {temp_script_path}"
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, timeout=timeout, encoding='utf-8')
        stderr_output = result.stderr.strip() if result.stderr else ""
        noisy_warnings = ["FutureWarning: You are using `torch.load`"]
        # Log warnings unless they are known noisy ones or debug messages
        if stderr_output and not any(warning in stderr_output for warning in noisy_warnings) and not stderr_output.startswith("DEBUG") and logger:
             logger.log_operation(**log_context, operation_type="extract", status="warning", details=f"Extraction script stderr: {stderr_output}")
        success = True
    except subprocess.TimeoutExpired as e:
         stderr_output = e.stderr.strip() if e.stderr else "N/A"; stdout_output = e.stdout.strip() if e.stdout else "N/A"
         error_details = f"Command timed out ({e.timeout}s): {e.cmd}\nStderr: {stderr_output}\nStdout: {stdout_output}"
         if logger: logger.log_operation(**log_context, operation_type="extract", status="failed", details=error_details)
    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.strip() if e.stderr else "N/A"; stdout_output = e.stdout.strip() if e.stdout else "N/A"
        error_details = f"Command failed (Exit Code {e.returncode}): {e.cmd}\nStderr: {stderr_output}\nStdout: {stdout_output}"
        if logger: logger.log_operation(**log_context, operation_type="extract", status="failed", details=error_details)
    except Exception as e:
        if logger: logger.log_operation(**log_context, operation_type="extract", status="failed", details=f"Unexpected error setting up/running temp script: {type(e).__name__}: {e}")
    finally:
        if os.path.exists(temp_script_path):
            try: os.unlink(temp_script_path)
            except OSError as unlink_e:
                 if logger: logger.log_operation(old_id=log_context.get('old_id','?'), new_id=log_context.get('new_id','?'), corpus_prefix=log_context.get('corpus_prefix','?'),source_file=temp_script_path, destination_file="", operation_type="cleanup", file_type="temp_script", status="failed", details=f"Could not remove temp script: {unlink_e}")
    # Log final success status explicitly
    if success and logger:
        # Don't log success here if only warnings were printed to stderr, handled above
        if not (stderr_output and not any(warning in stderr_output for warning in noisy_warnings)):
             logger.log_operation(**log_context, operation_type="extract", status="success")
    return success


# --- FileOperationLogger ---
class FileOperationLogger:
    """Logs operations to CSV and optionally WandB."""
    def __init__(self, log_file_path: str, use_wandb: bool = True, wandb_project: str = "corpus-reorganization"):
        self.log_file_path = log_file_path; self.log_entries = []; self.use_wandb = use_wandb; self.run_name = f"reorganization-{datetime.now().strftime('%Y%m%d-%H%M%S')}"; self.logger = logging.getLogger("FileLogger")
        try:
            log_dir = os.path.dirname(log_file_path);
            if log_dir: os.makedirs(log_dir, exist_ok=True)
            with open(log_file_path, 'w', newline='', encoding='utf-8') as csvfile: writer = csv.writer(csvfile); writer.writerow(['timestamp', 'old_id', 'new_id', 'corpus_prefix','source_file', 'destination_file', 'operation_type','file_type', 'status', 'details'])
            self.logger.info(f"CSV logging initialized at: {log_file_path}")
        except OSError as e: self.logger.error(f"Error creating log file {log_file_path}: {e}. Logging disabled for CSV."); self.log_file_path = None
        if use_wandb:
            try:
                wandb.init(project=wandb_project, name=self.run_name); wandb.config.update({"log_file_intended": log_file_path,"log_file_actual": self.log_file_path if self.log_file_path else "N/A","start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
                self.logger.info(f"WandB logging initialized for run: {wandb.run.name} (URL: {wandb.run.url if wandb.run else 'N/A'})")
            except Exception as e: self.logger.error(f"Error initializing wandb: {e}. Wandb logging disabled."); self.use_wandb = False

    def log_operation(self, status: str = "success", details: str = "", **log_ctx) -> None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S'); old_id = log_ctx.get('old_id', '?'); new_id = log_ctx.get('new_id', '?'); corpus_prefix = log_ctx.get('corpus_prefix', '?'); src_str = str(log_ctx.get('source_file', '')); dest_str = str(log_ctx.get('destination_file', '')); op_type = log_ctx.get('operation_type', '?'); file_type = log_ctx.get('file_type', '?'); details_str = str(details)
        entry = {'timestamp': timestamp,'old_id': old_id,'new_id': new_id,'corpus_prefix': corpus_prefix,'source_file': src_str,'destination_file': dest_str,'operation_type': op_type,'file_type': file_type,'status': status,'details': details_str}; self.log_entries.append(entry)
        log_level = logging.INFO;
        if status == "failed": log_level = logging.ERROR
        elif status == "warning": log_level = logging.WARNING
        log_msg = f"[{op_type}/{file_type}] ID:{old_id}->{new_id} Status:{status}";
        if src_str: log_msg += f" Src:{os.path.basename(src_str)}"
        if dest_str: log_msg += f" Dest:{os.path.basename(dest_str)}"
        if details_str: log_msg += f" Details: {details_str[:200]}{'...' if len(details_str)>200 else ''}"
        self.logger.log(log_level, log_msg)
        if self.log_file_path:
            try:
                with open(self.log_file_path, 'a', newline='', encoding='utf-8') as csvfile: writer = csv.writer(csvfile); writer.writerow([timestamp, old_id, new_id, corpus_prefix,src_str, dest_str, op_type,file_type, status, details_str])
            except OSError as e: self.logger.warning(f"Failed to write to CSV log {self.log_file_path}: {e}")
        if self.use_wandb and wandb.run:
            try:
                payload = {f"counts/status/{status}": 1, f"counts/operation/{op_type}": 1, f"counts/file_type/{file_type}": 1}
                if status == "failed": payload["errors/failure_count"] = 1; payload[f"errors/details/{op_type}_{file_type}"] = details_str[:500]
                wandb.log(payload, commit=True)
            except Exception as e: self.logger.warning(f"Failed to log operation to wandb: {e}")

    @timed_operation("summarize_logs")
    def summarize_and_close(self) -> Dict[str, Any]:
        total_operations = len(self.log_entries); status_counts = {'success': 0, 'failed': 0, 'warning': 0, 'info': 0, 'skipped': 0}; operation_counts = {}; file_type_counts = {}; unique_old_ids = set(); unique_new_ids = set()
        for entry in self.log_entries: status_counts[entry['status']] = status_counts.get(entry['status'], 0) + 1; op_type = entry['operation_type']; operation_counts[op_type] = operation_counts.get(op_type, 0) + 1; file_type = entry['file_type']; file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1;
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
            except Exception as e: self.logger.warning(f"Failed to log summary/finish wandb run: {e}")
        self.logger.info("Logger summarized and closed.")
        return summary


# --- Extraction Functions ---

# Integrates your enhanced main model extraction logic
@timed_operation("extract_main_outputs")
def extract_main_model_outputs(
    conda_env: str, docbin_path: str, output_txt_joined: str, output_txt_fullstop: str,
    output_csv_lemma: str, output_csv_upos: str, output_csv_stop: str, output_csv_dot: str,
    output_conllu: str, ner_tags_path: Optional[str], doc_id_str: str,
    logger: Optional['FileOperationLogger'] = None, **log_ctx
) -> bool:
    """Loads main model once and extracts all related outputs via one temp script, attempting NER alignment."""
    script_content = f"""# -*- coding: utf-8 -*-
import sys, csv, spacy, os, traceback, re, json
from difflib import SequenceMatcher # Added for alignment
from spacy.tokens import DocBin

# Removed global_stats dictionary - state managed by main process

docbin_path=r'{docbin_path}'; output_txt_joined=r'{output_txt_joined}'; output_txt_fullstop=r'{output_txt_fullstop}'; output_csv_lemma=r'{output_csv_lemma}'; output_csv_upos=r'{output_csv_upos}'; output_csv_stop=r'{output_csv_stop}'; output_csv_dot=r'{output_csv_dot}'; output_conllu=r'{output_conllu}'; ner_tags_path={repr(ner_tags_path)}; doc_id_str='{doc_id_str}'

# --- Alignment Function (Included from your suggestion) ---
def attempt_ner_alignment(tokens, ner_tags):
    token_texts = [str(t.text) for t in tokens]
    # Using SequenceMatcher to find longest common subsequences as blocks
    matcher = SequenceMatcher(None, token_texts, ner_tags, autojunk=False)
    aligned_tags = [None] * len(tokens)
    alignment_stats = {{'status': 'failed_no_matches', 'aligned_count': 0, 'total_tokens': len(tokens), 'total_ner_tags': len(ner_tags), 'success_rate': 0.0, 'details':[]}}
    blocks_processed = 0

    for block in matcher.get_matching_blocks():
        if block.size == 0: continue
        blocks_processed += 1
        token_start, ner_start, size = block
        for i in range(size):
            aligned_tags[token_start + i] = ner_tags[ner_start + i]
            alignment_stats['aligned_count'] += 1
            # Log limited details
            if len(alignment_stats['details']) < 10:
                 alignment_stats['details'].append(f"Align: tok[{{token_start+i}}]='{token_texts[token_start+i]}' <-> tag[{{ner_start+i}}]='{ner_tags[ner_start+i]}'")

    if alignment_stats['aligned_count'] > 0:
         alignment_stats['success_rate'] = alignment_stats['aligned_count'] / len(tokens) if tokens else 0
         alignment_stats['status'] = 'partial' if alignment_stats['aligned_count'] < len(tokens) else 'success'

    # Define a success threshold
    if alignment_stats['success_rate'] < 0.5: # Example threshold
        print(f"WARN(doc{{doc_id_str}}):Alignment quality low ({{alignment_stats['success_rate']:.1%}}). Discarding alignment.", file=sys.stderr)
        alignment_stats['status'] = 'failed_low_quality'
        return None, alignment_stats

    print(f"INFO(doc{{doc_id_str}}):Alignment Result: Status={{alignment_stats['status']}}, Aligned={{alignment_stats['aligned_count']}}/{{len(tokens)}} ({alignment_stats['success_rate']:.1%}).", file=sys.stderr)
    return aligned_tags, alignment_stats
# --- End Alignment Function ---

dirs_to_check=set([os.path.dirname(p) for p in [output_txt_joined,output_txt_fullstop,output_csv_lemma,output_csv_upos,output_csv_stop,output_csv_dot,output_conllu]])
for d in dirs_to_check:
    if d: os.makedirs(d, exist_ok=True)
try:
    nlp = spacy.load('grc_proiel_trf'); doc_bin = DocBin().from_disk(docbin_path); docs = list(doc_bin.get_docs(nlp.vocab)); assert docs, f"DocBin empty: {{docbin_path}}"
    doc = docs[0]; num_tokens = len(doc)
    doc_text = doc.text
    with open(output_txt_joined,'w',encoding='utf-8') as f: f.write(doc_text)
    try:
        # Fixed escape sequence warning
        tfs = re.sub(r'\\.(?!\\.)', '.\\n', doc_text); tfs = re.sub(r'\s+\\n','\\n', tfs); tfs = re.sub(r'\\n\s+', '\\n', tfs).strip()
        with open(output_txt_fullstop,'w',encoding='utf-8') as f: f.write(tfs)
    except Exception as fs_e: print(f"Warn: Fail fullstop: {{fs_e}}", file=sys.stderr)
    with open(output_csv_lemma,'w',encoding='utf-8',newline='') as fl,open(output_csv_upos,'w',encoding='utf-8',newline='') as fu,open(output_csv_stop,'w',encoding='utf-8',newline='') as fs,open(output_csv_dot,'w',encoding='utf-8',newline='') as fd:
        wl=csv.writer(fl,quoting=csv.QUOTE_ALL); wl.writerow(['ID','TOKEN','LEMMA']); wu=csv.writer(fu,quoting=csv.QUOTE_ALL); wu.writerow(['ID','TOKEN','UPOS']); ws=csv.writer(fs,quoting=csv.QUOTE_ALL); ws.writerow(['ID','TOKEN','IS_STOP']); wd=csv.writer(fd,quoting=csv.QUOTE_ALL); wd.writerow(['ID','TOKEN','IS_PUNCT'])
        for i,t in enumerate(doc): tid,ttxt=i+1,str(t.text); wl.writerow([tid,ttxt,t.lemma_]); wu.writerow([tid,ttxt,t.pos_]); ws.writerow([tid,ttxt,'TRUE' if t.is_stop else 'FALSE']); wd.writerow([tid,ttxt,'TRUE' if t.is_punct else 'FALSE'])

    # NER Processing & Alignment
    original_ner_tags = None; ner_tags_to_use = None; alignment_info = None; mismatch_detected = False
    if ner_tags_path and os.path.exists(ner_tags_path):
        try:
            with open(ner_tags_path,'r',encoding='utf-8') as fn: original_ner_tags=[ln.strip() for ln in fn if ln.strip()]
            if len(original_ner_tags) != num_tokens:
                mismatch_detected = True
                print(f"WARN(doc{{doc_id_str}}):Tok/NER mismatch! Main:{{num_tokens}} NER:{{len(original_ner_tags)}}. Attempting alignment.", file=sys.stderr)
                aligned_result, alignment_info = attempt_ner_alignment(doc, original_ner_tags)
                if aligned_result: ner_tags_to_use = aligned_result
                else: ner_tags_to_use = None; print(f"WARN(doc{{doc_id_str}}):Alignment failed/discarded, NER tags omitted from CoNLL-U.", file=sys.stderr)
            else: ner_tags_to_use = original_ner_tags # Counts match
        except Exception as e: print(f"Warn(doc{{doc_id_str}}):Fail read/align NER:{{e}}.", file=sys.stderr); ner_tags_to_use=None
    elif ner_tags_path: print(f"Warn(doc{{doc_id_str}}):NER tags path not found.", file=sys.stderr)

    # Write CoNLL-U
    with open(output_conllu,"w",encoding="utf-8") as fo:
        fo.write(f"# newdoc id = {{doc_id_str}}\\n")
        if mismatch_detected: # Add mismatch info comment
            fo.write(f"# ner_token_mismatch = True\\n"); fo.write(f"# main_model_tokens = {{num_tokens}}\\n"); fo.write(f"# ner_model_tags = {{len(original_ner_tags) if original_ner_tags else 'N/A'}}\\n")
            if alignment_info: fo.write(f"# ner_alignment_status = {{alignment_info.get('status','?')}}\\n"); fo.write(f"# ner_alignment_rate = {{alignment_info.get('success_rate',0):.4f}}\\n")
        sidc = 1
        for sent in doc.sents:
            stc=sent.text.replace('\\n',' ').replace('\\r',''); fo.write(f"# sent_id={{doc_id_str}}-{{sidc}}\\n"); fo.write(f"# text={{stc}}\\n")
            tsic=1
            for t in sent:
                hid=t.head.i-sent.start+1 if t.head.i!=t.i else 0; fts=str(t.morph) if t.morph and str(t.morph).strip() else "_"
                mp=[]
                if ner_tags_to_use: # Use potentially aligned or original tags
                    nt = ner_tags_to_use[t.i] # Access using token index; will be None if unaligned
                    if nt and nt!='O': mp.append(f"NER={{nt}}")
                if t.i+1<num_tokens and doc[t.i+1].idx==t.idx+len(t.text): mp.append("SpaceAfter=No")
                mf="|".join(mp) if mp else "_"; dpf="_"; dpr=str(t.dep_) if t.dep_ and t.dep_.strip() else "dep"
                cols=[str(tsic),str(t.text),str(t.lemma_),str(t.pos_),str(t.tag_),fts,str(hid),dpr,dpf,mf]; fo.write("\\t".join(cols)+"\\n"); tsic+=1
            fo.write("\\n"); sidc+=1

    # Write mismatch details file if mismatch occurred
    if mismatch_detected:
        mismatch_info_file = os.path.join(os.path.dirname(output_conllu), f"{{doc_id_str}}_ner_mismatch_info.json")
        mismatch_data = {{ "document_id": doc_id_str, "main_model_tokens": num_tokens, "ner_model_tags": len(original_ner_tags) if original_ner_tags else None, "mismatch_detected": mismatch_detected, "alignment_info": alignment_info }}
        try:
            with open(mismatch_info_file, 'w', encoding='utf-8') as fm: json.dump(mismatch_data, fm, indent=2)
        except Exception as e: print(f"Failed write mismatch info: {{e}}", file=sys.stderr)

except Exception as e: print(f"Error: {{e}}",file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.exit(1)"""
    log_ctx.update({'source_file': docbin_path, 'destination_file': f"Multiple files in {os.path.dirname(output_txt_joined)} and {os.path.dirname(output_csv_lemma)}", 'file_type': 'main-model-outputs'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger, timeout=1200) # Increased timeout

# Integrates your enhanced NER model extraction
@timed_operation("extract_ner_outputs")
def extract_ner_model_outputs(
    conda_env: str, docbin_path: str, output_csv_ner: str, output_ner_tags: str,
    logger: Optional['FileOperationLogger'] = None, **log_ctx
) -> bool:
    """Loads NER model once and extracts NER CSV and NER tags file via one temp script."""
    # *** Verify this model name is correct for your NER environment ***
    ner_model_name = 'grc_ner_trf'
    script_content = f"""# -*- coding: utf-8 -*-
import sys, csv, spacy, os, traceback, json; from spacy.tokens import DocBin
ner_stats={{'total_tokens':0,'ner_tokens':0,'o_tokens':0}}
docbin_path=r'{docbin_path}'; output_csv_ner=r'{output_csv_ner}'; output_ner_tags=r'{output_ner_tags}'; ner_model_name='{ner_model_name}'
dirs_to_check=set([os.path.dirname(p) for p in [output_csv_ner, output_ner_tags]]); [os.makedirs(d, exist_ok=True) for d in dirs_to_check if d]
try:
    nlp = spacy.load(ner_model_name); doc_bin = DocBin().from_disk(docbin_path); docs = list(doc_bin.get_docs(nlp.vocab)); assert docs, f"DocBin empty: {{docbin_path}}"
    doc = docs[0]; num_tokens = len(doc); ner_stats['total_tokens']=num_tokens
    print(f"DEBUG NER: Doc {log_ctx.get('old_id','?')}, Loaded '{os.path.basename(docbin_path)}' w/ {ner_model_name}. Found {{len(doc.ents)}} ents.", file=sys.stderr)
    with open(output_csv_ner,'w',encoding='utf-8',newline='') as fner, open(output_ner_tags,'w',encoding='utf-8') as ftags:
        wn=csv.writer(fner,quoting=csv.QUOTE_ALL); wn.writerow(['ID','TOKEN','NER'])
        tags_list = []
        for i,t in enumerate(doc):
            tid,ttxt=i+1,str(t.text); nt=t.ent_type_ if t.ent_type_ else 'O'
            if nt!='O': ner_stats['ner_tokens']+=1
            else: ner_stats['o_tokens']+=1
            wn.writerow([tid, ttxt, nt]); tags_list.append(nt)
        ftags.write('\\n'.join(tags_list))
    ner_summary_file = os.path.join(os.path.dirname(output_csv_ner), f"{log_ctx.get('new_id','?')}_ner_stats.json") # Use new_id
    try:
        with open(ner_summary_file,'w',encoding='utf-8') as fsum: json.dump({{ "doc_id": "{log_ctx.get('old_id','?')}", "new_id": "{log_ctx.get('new_id','?')}", "docbin_path": docbin_path, "total_tokens": ner_stats['total_tokens'], "tokens_with_ner": ner_stats['ner_tokens'], "tokens_without_ner": ner_stats['o_tokens'], "ner_percentage": ner_stats['ner_tokens']/ner_stats['total_tokens'] if ner_stats['total_tokens']>0 else 0 }}, fsum, indent=2)
    except Exception as e: print(f"Failed write NER summary: {{e}}", file=sys.stderr)
    print(f"NER OK: Total={{num_tokens}}, NER={{ner_stats['ner_tokens']}}", file=sys.stderr)
except Exception as e: print(f"Error: {{e}}",file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.exit(1)"""
    log_ctx.update({'source_file': docbin_path, 'destination_file': f"{os.path.basename(output_csv_ner)} & {os.path.basename(output_ner_tags)}", 'file_type': 'ner-model-outputs'})
    return run_python_script_in_conda_env(conda_env, script_content, log_ctx, logger, timeout=600)


# --- Function to summarize mismatches (Added from your suggestion) ---
@timed_operation("summarize_mismatches")
def summarize_token_mismatches(output_base_dir: str, logger: Optional[FileOperationLogger] = None):
    """Collects and summarizes token mismatch information across the corpus."""
    # Use Pathlib for better path handling
    base_path = Path(output_base_dir)
    mismatch_files = list(base_path.glob("**/*_ner_mismatch_info.json"))
    script_logger.info(f"Found {len(mismatch_files)} mismatch info files to summarize in {output_base_dir}.")
    if not mismatch_files: return None

    total_docs_checked = 0; mismatched_docs_count = 0; alignment_attempts = 0; alignment_successes = 0;
    token_diff_stats = Counter(); alignment_statuses = Counter()

    for mismatch_file in mismatch_files:
        total_docs_checked += 1
        try:
            with open(mismatch_file, 'r', encoding='utf-8') as f: data = json.load(f)
            if data.get('mismatch_detected', False): mismatched_docs_count += 1
            alignment_info = data.get('alignment_info')
            if alignment_info:
                 alignment_attempts += 1
                 status = alignment_info.get('status')
                 if status: alignment_statuses[status] += 1
                 # Consider 'partial' as success for rate calculation? Or only 'success'? Let's count 'partial' too.
                 if status in ('partial', 'success'): alignment_successes += 1
            main_tokens = data.get('main_model_tokens'); ner_tokens = data.get('ner_model_tags')
            if main_tokens is not None and ner_tokens is not None:
                 token_diff_stats[ner_tokens - main_tokens] += 1
        except Exception as e: script_logger.warning(f"Error processing mismatch file {mismatch_file}: {e}")

    mismatch_percentage = (mismatched_docs_count / total_docs_checked * 100) if total_docs_checked else 0
    alignment_success_rate = (alignment_successes / alignment_attempts * 100) if alignment_attempts else 0

    summary = {
        "total_documents_with_mismatch_file": total_docs_checked,
        "mismatched_documents_found": mismatched_docs_count,
        "mismatch_percentage": mismatch_percentage,
        "alignment_attempts": alignment_attempts,
        "alignment_successes_incl_partial": alignment_successes,
        "alignment_success_rate_incl_partial": alignment_success_rate,
        "alignment_status_counts": dict(alignment_statuses),
        "token_difference_distribution": {str(k): v for k, v in sorted(token_diff_stats.items())}
    }
    summary_file = base_path / "corpus_token_mismatch_summary.json"
    try:
        with open(summary_file, 'w', encoding='utf-8') as f: json.dump(summary, f, indent=2)
        script_logger.info(f"Token mismatch summary written to: {summary_file}")
        script_logger.info(f"Found {mismatched_docs_count} mismatched docs ({summary['mismatch_percentage']:.2f}%). Alignment success rate (incl. partial): {summary['alignment_success_rate_incl_partial']:.2f}% ({alignment_successes}/{alignment_attempts})")
        if logger: logger.log_operation(old_id='corpus',new_id='summary',corpus_prefix='all',operation_type='mismatch_summary',file_type='json',source_file='multiple',destination_file=str(summary_file),status='success',details=f"Mismatched:{mismatched_docs_count}/{total_docs_checked}, Align Success:{alignment_successes}/{alignment_attempts}")
    except Exception as e:
        script_logger.error(f"Failed to write token mismatch summary: {e}")
        if logger: logger.log_operation(old_id='corpus',new_id='summary',corpus_prefix='all',operation_type='mismatch_summary',file_type='json',source_file='multiple',destination_file=str(summary_file),status='failed',details=f"Error: {e}")
    return summary


# --- Helper Functions ---
def validate_input_file(file_path: str, file_type: str) -> bool:
    """Checks if an input file exists and is readable."""
    if not file_path or not os.path.exists(file_path): script_logger.error(f"Input {file_type} file not found: {file_path}"); return False
    if not os.path.isfile(file_path): script_logger.error(f"Input {file_type} path is not a file: {file_path}"); return False
    if not os.access(file_path, os.R_OK): script_logger.error(f"Input {file_type} file not readable: {file_path}"); return False
    return True

def validate_output_dir(dir_path: str) -> bool:
    """Checks if output directory exists or can be created."""
    if not dir_path: script_logger.error("Output directory path cannot be empty."); return False
    try:
        os.makedirs(dir_path, exist_ok=True); test_file = os.path.join(dir_path, ".perm_check")
        with open(test_file, "w") as f: f.write("test"); os.remove(test_file); return True
    except OSError as e: script_logger.error(f"Cannot create/write to output dir '{dir_path}': {e}"); return False

@timed_operation("load_index")
def load_index(index_path: str) -> Dict[str, str]:
    index = {}
    try:
        df = pd.read_csv(index_path); required_path_col = 'processed_path'
        if 'document_id' in df.columns and required_path_col in df.columns:
            for _, row in df.iterrows():
                doc_id = str(row['document_id']).strip(); file_path = str(row[required_path_col]).strip()
                if doc_id and file_path: index[doc_id] = file_path
        else: logging.warning(f"Index file '{index_path}' missing 'document_id' or '{required_path_col}'.")
    except FileNotFoundError: logging.error(f"Index file not found at '{index_path}'"); return {}
    except Exception as e: logging.error(f"Error loading index file '{index_path}': {e}")
    return index

@timed_operation("parse_mapping")
def parse_csv_mapping(csv_path: str) -> Dict[str, str]:
    mappings = {};
    try:
        df = pd.read_csv(csv_path)
        if 'document_id' in df.columns and 'sort_id' in df.columns:
             for index, row in df.iterrows():
                old_id = str(row['document_id']).strip(); new_id_val = row['sort_id']
                if pd.notna(new_id_val):
                    try: new_id = str(int(new_id_val))
                    except ValueError: new_id = None; logging.warning(f"Invalid sort_id '{new_id_val}' for doc '{old_id}'. Skip.")
                else: new_id = None
                if old_id and new_id: mappings[old_id] = new_id
                elif old_id and new_id is None and pd.isna(row['sort_id']): logging.warning(f"Missing sort_id for doc '{old_id}'. Skip.")
        else: logging.warning(f"Mapping file '{csv_path}' missing 'document_id' or 'sort_id'.")
    except FileNotFoundError: logging.error(f"Mapping file not found at '{csv_path}'"); return {}
    except Exception as e: logging.error(f"Error parsing mapping file '{csv_path}': {e}")
    return mappings

def find_original_text(doc_id: str, base_dir: str) -> Optional[str]:
    corpus_prefix_match = re.match(r'^([a-zA-Z0-9\-_]+?)_', doc_id); corpus_prefix = corpus_prefix_match.group(1) if corpus_prefix_match else ""
    source_sub_dir = "cleaned_parsed_data"; possible_paths = []
    if corpus_prefix:
        cleaned_dir = os.path.join(base_dir, source_sub_dir, corpus_prefix); filename_part = doc_id.replace(f"{corpus_prefix}_", "", 1)
        possible_paths.extend([os.path.join(cleaned_dir, filename_part + ".txt"), os.path.join(cleaned_dir, doc_id + ".txt")])
    possible_paths.append(os.path.join(base_dir, source_sub_dir, doc_id + ".txt"))
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path) and os.path.isfile(abs_path): return abs_path
    return None


# --- Main Processing Logic ---
@timed_operation("process_document")
def process_document(
    old_id: str, new_id: str, base_dir: str, output_base_dir: str,
    main_index: Dict[str, str], ner_index: Dict[str, str],
    main_env: str, ner_env: str, logger: Optional[FileOperationLogger] = None,
    overwrite: bool = False, use_locking: bool = False
) -> Tuple[bool, str]:
    overall_success = True; final_status_msg = "Success"; temp_files_to_clean = []
    corpus_prefix_match = re.match(r'^([a-zA-Z0-9\-_]+?)_', old_id); corpus_prefix = corpus_prefix_match.group(1) if corpus_prefix_match else "unknown_corpus"
    log_context_base = {'old_id': old_id, 'new_id': new_id, 'corpus_prefix': corpus_prefix}
    if logger: logger.log_operation(**log_context_base, operation_type="process_start", file_type="document", status="info", details=f"Started (O={overwrite}, L={use_locking})") # Shortened details

    main_docbin_path = main_index.get(old_id); ner_docbin_path = ner_index.get(old_id); ner_docbin_path_resolved = None
    if ner_docbin_path and os.path.exists(ner_docbin_path) and os.path.isfile(ner_docbin_path): ner_docbin_path_resolved = ner_docbin_path
    elif ner_docbin_path and logger: logger.log_operation(**log_context_base, operation_type="lookup", file_type="ner_docbin", status="warning", details="NER index path invalid/not found.")
    source_txt_path = find_original_text(old_id, base_dir)

    output_dir = os.path.join(output_base_dir, corpus_prefix, new_id); texts_dir = os.path.join(output_dir, "texts"); annotations_dir = os.path.join(output_dir, "annotations")
    try: os.makedirs(texts_dir, exist_ok=True); os.makedirs(annotations_dir, exist_ok=True)
    except OSError as e:
         if logger: logger.log_operation(**log_context_base, operation_type="create_dir", file_type="directory", status="failed", details=f"Error: {e}")
         return False, f"Failed: Dir creation error {e}"

    # Define output paths
    output_txt_joined=os.path.join(texts_dir,f"{new_id}-joined.txt"); output_txt_fullstop=os.path.join(texts_dir,f"{new_id}-fullstop.txt")
    output_csv_lemma=os.path.join(annotations_dir,f"{new_id}-lemma.csv"); output_csv_upos=os.path.join(annotations_dir,f"{new_id}-upos.csv")
    output_csv_stop=os.path.join(annotations_dir,f"{new_id}-stop.csv"); output_csv_dot=os.path.join(annotations_dir,f"{new_id}-dot.csv")
    output_csv_ner=os.path.join(annotations_dir,f"{new_id}-ner.csv"); output_conllu=os.path.join(annotations_dir,f"{new_id}-conllu.conllu")
    temp_ner_tags_file=Path(annotations_dir)/f".{new_id}_temp_ner_tags.txt"; temp_files_to_clean.append(str(temp_ner_tags_file))
    ner_tags_lock_path=str(temp_ner_tags_file)+".lock"
    main_model_files=[output_txt_joined,output_txt_fullstop,output_csv_lemma,output_csv_upos,output_csv_stop,output_csv_dot,output_conllu]
    ner_model_files=[output_csv_ner,str(temp_ner_tags_file)]

    try:
        ner_tags_available = False; ner_step_skipped = False
        if ner_docbin_path_resolved:
            ner_outputs_exist = all(os.path.exists(f) for f in ner_model_files)
            if overwrite or not ner_outputs_exist:
                logging.debug(f"Extracting NER outputs for {new_id}...")
                ner_lock = filelock.FileLock(ner_tags_lock_path, timeout=10) if use_locking and FILELOCK_AVAILABLE else filelock.FileLock(ner_tags_lock_path, timeout=-1)
                try:
                    with ner_lock.acquire():
                        if extract_ner_model_outputs(ner_env, ner_docbin_path_resolved, output_csv_ner, str(temp_ner_tags_file), logger, **log_context_base): ner_tags_available = True
                        else: overall_success = False; final_status_msg = "Failed: NER extraction"
                except filelock.Timeout:
                    if logger: logger.log_operation(**log_context_base, operation_type="lock", file_type="ner-tags", status="failed", details=f"Timeout lock {os.path.basename(ner_tags_lock_path)}")
                    overall_success = False; final_status_msg = "Failed: NER lock timeout"
                except Exception as lock_e:
                     if logger: logger.log_operation(**log_context_base, operation_type="lock", file_type="ner-tags", status="failed", details=f"Error lock {os.path.basename(ner_tags_lock_path)}: {lock_e}")
                     overall_success = False; final_status_msg = "Failed: NER lock error"
            elif logger: logger.log_operation(**log_context_base, operation_type="extract", file_type="ner-model-outputs", status="skipped", details="All output files exist"); ner_tags_available = True; ner_step_skipped = True
        else:
             if logger: logger.log_operation(**log_context_base, operation_type="extract", file_type="ner-model-outputs", status="skipped", details="NER docbin not resolved"); ner_step_skipped = True

        if overall_success: # Only proceed if NER step didn't critically fail
            main_outputs_exist = all(os.path.exists(f) for f in main_model_files)
            if overwrite or not main_outputs_exist:
                ner_tags_input_path = None; ner_lock = filelock.FileLock(ner_tags_lock_path, timeout=10) if use_locking and FILELOCK_AVAILABLE else filelock.FileLock(ner_tags_lock_path, timeout=-1)
                try:
                    with ner_lock.acquire():
                        if os.path.exists(str(temp_ner_tags_file)): ner_tags_input_path = str(temp_ner_tags_file)
                        # Removed redundant logging here, handled by function success/fail
                    logging.info(f"Extracting Main Model outputs for {new_id} (using NER tags: {ner_tags_input_path is not None})...")
                    if not extract_main_model_outputs(main_env, main_docbin_path, output_txt_joined, output_txt_fullstop, output_csv_lemma, output_csv_upos, output_csv_stop, output_csv_dot, output_conllu, ner_tags_input_path, new_id, logger, **log_context_base):
                        overall_success = False; final_status_msg = "Failed: Main model extraction"
                except filelock.Timeout:
                    if logger: logger.log_operation(**log_context_base, operation_type="lock", file_type="conllu-ner-read", status="failed", details=f"Timeout lock {os.path.basename(ner_tags_lock_path)}")
                    overall_success = False; final_status_msg = "Failed: CoNLLU lock timeout"
                except Exception as lock_e:
                     if logger: logger.log_operation(**log_context_base, operation_type="lock", file_type="conllu-ner-read", status="failed", details=f"Error lock {os.path.basename(ner_tags_lock_path)}: {lock_e}")
                     overall_success = False; final_status_msg = "Failed: CoNLLU lock error"
            elif logger: logger.log_operation(**log_context_base, operation_type="extract", file_type="main-model-outputs", status="skipped", details="All output files exist")

        dest_original_txt = os.path.join(texts_dir, f"{new_id}-original.txt")
        if source_txt_path:
            if overwrite or not os.path.exists(dest_original_txt):
                try: shutil.copy2(source_txt_path, dest_original_txt);
                except Exception as e:
                    if logger: logger.log_operation(**log_context_base, operation_type="copy", file_type="source_txt", status="failed", details=str(e))
            elif logger: logger.log_operation(**log_context_base, operation_type="copy", file_type="source_txt", status="skipped", details="Destination exists")

    finally:
        for temp_file_path in temp_files_to_clean:
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except OSError as e:
                    if logger: logger.log_operation(**log_context_base, operation_type="cleanup", file_type="temp_file", status="failed", details=f"Error: {e}")
        if use_locking and FILELOCK_AVAILABLE and os.path.exists(ner_tags_lock_path):
            try: os.remove(ner_tags_lock_path)
            except OSError as e:
                 if logger: logger.log_operation(**log_context_base, operation_type="cleanup", file_type="lock_file", status="failed", details=f"Failed remove lock {os.path.basename(ner_tags_lock_path)}: {e}")

    if logger: logger.log_operation(**log_context_base, operation_type="process_end", file_type="document", status="success" if overall_success else "failed", details=f"Finished. Status: {final_status_msg}")
    return overall_success, final_status_msg


# --- (#8) Graceful Shutdown Handler ---
_shutdown_requested = False; _executor_ref = None
def setup_graceful_shutdown():
    global _executor_ref
    def handler(sig, frame):
        global _shutdown_requested, _executor_ref
        if not _shutdown_requested:
            msg = f"Shutdown requested (Signal {sig}). Waiting for active tasks..."; logging.warning(msg); script_logger.warning(msg); print(f"\n{msg}", file=sys.stderr)
            _shutdown_requested = True
            if _executor_ref:
                logging.info("Requesting executor shutdown (wait=True)...")
                try: _executor_ref.shutdown(wait=True, cancel_futures=False)
                except Exception as e: logging.error(f"Error during executor shutdown: {e}")
                logging.info("Executor shutdown complete.")
            sys.exit(0)
        else: logging.info("Shutdown already in progress.")
    signal.signal(signal.SIGINT, handler); signal.signal(signal.SIGTERM, handler)


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize corpus documents...")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel processes (Default: 1).")
    parser.add_argument("--use-locking", action="store_true", help=f"Use file locking for parallel safety (Requires 'filelock': {'Available' if FILELOCK_AVAILABLE else 'NOT FOUND'}).")
    parser.add_argument("--mapping-csv", required=True, help="Path to CSV mapping 'document_id' to 'sort_id'.")
    parser.add_argument("--main-index-csv", required=True, help="Path to CSV index for main DocBins ('processed_path').")
    parser.add_argument("--ner-index-csv", required=True, help="Path to CSV index for NER DocBins ('processed_path').")
    parser.add_argument("--base-dir", required=True, help="Path to base directory containing original source data.")
    parser.add_argument("--output-dir", required=True, help="Path to base directory for reorganized output.")
    parser.add_argument("--main-env", required=True, help="Name of Conda env for main model.")
    parser.add_argument("--ner-env", required=True, help="Name of Conda env for NER model.")
    parser.add_argument("--log-file", default="reorganization_log.csv", help="Path for CSV log file.")
    parser.add_argument("--wandb-project", default="corpus-reorganization", help="WandB project name.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging.")
    args = parser.parse_args()

    # (#11) Input Validation
    valid_inputs = True
    if not validate_input_file(args.mapping_csv, "Mapping CSV"): valid_inputs = False
    if not validate_input_file(args.main_index_csv, "Main Index CSV"): valid_inputs = False
    if not validate_input_file(args.ner_index_csv, "NER Index CSV"): valid_inputs = False
    if not os.path.isdir(args.base_dir): logging.warning(f"Base directory '{args.base_dir}' does not exist.")
    if not validate_output_dir(args.output_dir): valid_inputs = False
    if args.use_locking and not FILELOCK_AVAILABLE: logging.warning("--use-locking specified but 'filelock' not found. Disabling locking."); args.use_locking = False
    if not valid_inputs: script_logger.error("Invalid inputs. Exiting."); sys.exit(1)

    start_time = time.time()
    logger = FileOperationLogger(log_file_path=args.log_file, use_wandb=(not args.no_wandb), wandb_project=args.wandb_project)

    script_logger.info(f"Starting. Overwrite:{args.overwrite}, Workers:{args.num_workers}, Locking:{args.use_locking}")
    script_logger.info("-" * 30); [script_logger.info(f"{arg:<20}: {value}") for arg, value in vars(args).items()]; script_logger.info("-" * 30)

    if os.path.abspath(args.main_index_csv) == os.path.abspath(args.ner_index_csv): logging.warning("="*60); logging.warning("Warn: Main & NER index files same."); logging.warning(" NER likely only 'O' tags."); logging.warning("="*60)

    if logger.use_wandb and wandb.run:
         try:
              args_for_config=vars(args).copy(); current_wandb_log_file_actual=wandb.config.get("log_file_actual")
              if "log_file" in args_for_config and current_wandb_log_file_actual is not None: del args_for_config["log_file"]
              wandb.config.update(args_for_config, allow_val_change=True)
         except Exception as e: logging.warning(f"Failed update wandb config: {e}")

    script_logger.info("Loading mappings and indices...")
    mappings = parse_csv_mapping(args.mapping_csv)
    main_index = load_index(args.main_index_csv)
    ner_index = load_index(args.ner_index_csv)
    script_logger.info(f"Loaded {len(mappings)} maps, {len(main_index)} main idx, {len(ner_index)} NER idx.")
    if not mappings or not main_index: script_logger.error("Mapping/Main Index empty/failed. Exiting."); exit(1)

    tasks = []; skipped_count = 0
    for old_id, new_id in mappings.items():
        main_docbin_path_check = main_index.get(old_id)
        if not main_docbin_path_check or not os.path.exists(main_docbin_path_check):
             if logger: logger.log_operation(old_id=old_id, new_id=new_id, corpus_prefix=(old_id.split('_')[0] if '_' in old_id else '?'), operation_type="lookup", file_type="main_docbin", status="skipped", details="Not found in index or path invalid.")
             skipped_count += 1; continue
        tasks.append({'old_id': old_id, 'new_id': new_id, 'base_dir': args.base_dir, 'output_base_dir': args.output_dir,'main_index': main_index, 'ner_index': ner_index, 'main_env': args.main_env, 'ner_env': args.ner_env,'logger': logger, 'overwrite': args.overwrite, 'use_locking': args.use_locking})

    processed_count = len(tasks); failed_count = 0
    script_logger.info(f"Processing {processed_count} documents ({skipped_count} skipped)...")

    def process_document_wrapper(task_args):
        try: return process_document(**task_args)
        except Exception as worker_exc:
            worker_old_id = task_args.get('old_id', 'unknown'); script_logger.error(f"Exception in worker for {worker_old_id}: {worker_exc}"); traceback.print_exc()
            return False, f"Failed: Worker Exception {type(worker_exc).__name__}"

    executor = None; pbar = None
    setup_graceful_shutdown()

    try:
        if args.num_workers > 1 and processed_count > 0 :
            script_logger.info(f"Starting parallel processing with {args.num_workers} workers.")
            pbar = tqdm.tqdm(total=processed_count, desc="Processing Docs", unit="doc", dynamic_ncols=True)
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                _executor_ref = executor
                futures = {executor.submit(process_document_wrapper, task): task for task in tasks}
                for future in as_completed(futures):
                    task_info = futures[future]; task_id_str = f"{task_info.get('old_id','?')}->{task_info.get('new_id','?')}"
                    if _shutdown_requested: break
                    try: success, final_status_msg = future.result();
                         if not success: failed_count += 1
                         pbar.set_postfix_str(f"Last: {task_id_str}, Status: {final_status_msg}", refresh=True)
                    except Exception as exc: script_logger.error(f'Task {task_id_str} generated exception: {exc}'); traceback.print_exc(); failed_count += 1; pbar.set_postfix_str(f"Last: {task_id_str}, Status: Exception", refresh=True)
                    finally: pbar.update(1)
            _executor_ref = None # Clear before implicit shutdown
        else:
            script_logger.info("Starting sequential processing.")
            pbar = tqdm.tqdm(tasks, desc="Processing Docs", unit="doc", dynamic_ncols=True)
            for task in pbar:
                if _shutdown_requested: script_logger.warning("Shutdown requested, stopping."); break
                task_id_str = f"{task.get('old_id','?')}->{task.get('new_id','?')}"; pbar.set_postfix_str(f"Current: {task_id_str}", refresh=False)
                success, final_status_msg = process_document_wrapper(task)
                if not success: failed_count += 1
                pbar.set_postfix_str(f"Last: {task_id_str}, Status: {final_status_msg}", refresh=True)
    except KeyboardInterrupt: script_logger.warning("KeyboardInterrupt received. Exiting.")
        if _executor_ref: _executor_ref.shutdown(wait=False, cancel_futures=True)
    finally:
        if pbar: pbar.close()

    # --- Run Mismatch Summary ---
    if processed_count > 0: # Only summarize if files were likely created
        summarize_token_mismatches(args.output_dir, logger)

    # Final Summary... (same as before)
    end_time = time.time(); duration = end_time - start_time; attempted_processing = processed_count
    summary_msg = ("\n"+"-"*30 + "\n--- Reorganization Summary ---" + f"\nDocuments in mapping file: {len(mappings)}" +
                   f"\nDocuments skipped (missing main docbin): {skipped_count}" + f"\nDocuments attempted processing: {attempted_processing}" +
                   f"\n-> Successfully processed: {attempted_processing - failed_count}" + f"\n-> Failed processing: {failed_count}" +
                   f"\nTotal time taken: {duration:.2f} seconds ({duration/60:.2f} minutes)" + "\n" + "-"*30)
    script_logger.info(summary_msg); print(summary_msg)
    summary_stats = logger.summarize_and_close()
    if logger.log_file_path: script_logger.info(f"Detailed log saved to: {logger.log_file_path}")
    if not args.no_wandb:
        if wandb.run and wandb.run.url: script_logger.info(f"W&B Run URL: {wandb.run.url}")
        else: script_logger.info("W&B run finished or was not available.")
    script_logger.info("Script finished."); print("\nScript finished.")
