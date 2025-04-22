# -*- coding: utf-8 -*-
"""
Script responsible for cleaning/processing a corpus using spaCy.

Loads a single spaCy model and processes documents sequentially.
Uses spaCy's nlp.pipe() for parallelism when processing segments
of documents that exceed the configured max_length.
Logs progress, system metrics, and NLP statistics to Weights & Biases.
Retains ALL token attributes as requested.
Includes optional single-file test mode.
"""
import argparse
import gc
import glob
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import plotly.graph_objects as go
import psutil
import spacy
import torch
import wandb
from spacy.language import Language
from spacy.tokens import Doc, DocBin
from tqdm import tqdm

# Import helper without requiring it to be a module installation
# Assumes utils/streams.py is in a place Python can find it (e.g., PYTHONPATH or same dir)
try:
    from utils.streams import stream_files
except ImportError:
    print("Error: Could not import stream_files from utils.streams.")
    print("Ensure utils/streams.py is accessible (e.g., in the same directory or PYTHONPATH).")
    sys.exit(1)

from wandb.data_types import Plotly

# --- NUMA/Resource Awareness ---
# (NUMA import logic unchanged)
try:
    import numexpr
except ImportError:
    numexpr = None

try:
    import numa
    _ = numa.num_configured_nodes() # Basic check
    print("NUMA library ('numa') imported successfully.")
except ImportError:
    print("NUMA Python library (e.g., 'numa') not installed. NUMA awareness disabled.")
    print("Suggestion: [sudo apt install libnuma-dev &&] pip install numa")
    numa = None
except Exception as e:
    print(f"NUMA library loaded but failed initial check (NUMA likely not supported by OS/hardware): {e}")
    numa = None


# --- Configuration ---
# (RunConfig and SystemInfo dataclasses unchanged)
@dataclass
class SystemInfo:
    """Detected system resource information."""
    cpu_count: int
    available_memory_gb: float
    numa_nodes: int = 1

@dataclass
class RunConfig:
    """Configuration for the processing run, combining CLI args and system info."""
    # --- Paths ---
    src_index: Path
    dest_dir: Path
    # --- Model ---
    model_name: str
    # --- Performance Tuning (Defaults set based on SystemInfo later) ---
    max_length: int = 0 # Document character limit before splitting
    n_process: int = 0 # Number of processes for nlp.pipe()
    batch_size: int = 0 # Batch size for nlp.pipe()
    torch_threads: int = 0 # Threads for PyTorch (main process)
    # --- Execution Control ---
    force_sync: bool = False # DANGEROUSLY SLOW
    sample_interval: int = 50 # Log detailed stats every N docs
    checkpoint_interval: int = 200 # Save checkpoint every N docs
    test_doc_id: Optional[str] = None # <<< ADDED for test mode
    test_src_path: Optional[str] = None # <<< ADDED for test mode
    # --- W&B ---
    wandb_user: Optional[str] = None
    wandb_project: Optional[str] = None
    run_name: Optional[str] = None
    disable_wandb: bool = False
    # --- System Info (populated after init) ---
    system_info: SystemInfo = field(init=False)
    # --- Internal ---
    _run_id: str = field(default_factory=lambda: f"{time.strftime('%Y%m%d-%H%M%S')}")

    def __post_init__(self):
        # (post_init unchanged)
        self.system_info = self._detect_system()
        defaults = self._calculate_defaults(self.system_info)
        if self.max_length <= 0: self.max_length = defaults['max_length']
        if self.n_process <= 0: self.n_process = defaults['n_process']
        if self.batch_size <= 0: self.batch_size = defaults['batch_size']
        if self.torch_threads <= 0: self.torch_threads = defaults['torch_threads']
        if not self.run_name: self.run_name = f"corpus-proc-{self.model_name}-{self._run_id}"
        self.src_index = Path(self.src_index)
        self.dest_dir = Path(self.dest_dir)
        if self.test_src_path: self.test_src_path = Path(self.test_src_path) # Convert test path too


    def _detect_system(self) -> SystemInfo:
        # (unchanged)
        cpu_count = os.cpu_count() or 1
        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
        numa_nodes_detected = 1
        if numa:
            try:
                nodes = numa.num_configured_nodes()
                if nodes > 0: numa_nodes_detected = nodes
            except Exception: pass
        return SystemInfo(cpu_count, available_memory_gb, numa_nodes_detected)

    def _calculate_defaults(self, sys_info: SystemInfo) -> Dict[str, int]:
        # (unchanged)
        n_process = max(1, min(32, int(sys_info.cpu_count * 0.50)))
        batch_size = max(16, min(int(sys_info.available_memory_gb * 3), 128))
        max_length = min(1_000_000, max(100_000, int(sys_info.available_memory_gb * 10 * 10**3)))
        torch_threads = max(2, min(16, sys_info.cpu_count // 2))
        print("--- Calculated Default Performance Settings ---")
        print(f"  Defaults: nlp.pipe processes = {n_process}")
        print(f"            nlp.pipe batch_size = {batch_size}")
        print(f"            max_length = {max_length}")
        print(f"            torch_threads = {torch_threads}")
        print("  (These can be overridden by CLI arguments)")
        print("---------------------------------------------")
        return {'n_process': n_process, 'batch_size': batch_size, 'max_length': max_length, 'torch_threads': torch_threads}

    def log_to_wandb(self):
        # (unchanged)
        if not self.disable_wandb and wandb.run:
            config_dict = { "model_name": self.model_name, "src_index": str(self.src_index), "dest_dir": str(self.dest_dir),
                "max_length": self.max_length, "n_process_pipe": self.n_process, "batch_size_pipe": self.batch_size,
                "torch_threads": self.torch_threads, "force_sync": self.force_sync, "sample_interval": self.sample_interval,
                "checkpoint_interval": self.checkpoint_interval, "system_cpu_count": self.system_info.cpu_count,
                "system_available_memory_gb": self.system_info.available_memory_gb, "system_numa_nodes": self.system_info.numa_nodes,
                "numa_library_present": numa is not None, "run_id": self._run_id,
                "test_mode_doc_id": self.test_doc_id } # Log if test mode was used
            wandb.config.update(config_dict)

# --- Constants ---
# (TOKEN_ATTRS unchanged)
TOKEN_ATTRS = [
    "IS_ALPHA", "IS_ASCII", "IS_DIGIT", "IS_LOWER", "IS_PUNCT", "IS_SPACE", "IS_TITLE", "IS_UPPER", "LIKE_URL", "LIKE_NUM", "LIKE_EMAIL", "IS_STOP",
    "IS_QUOTE", "IS_BRACKET", "IS_LEFT_PUNCT", "IS_RIGHT_PUNCT", "IS_CURRENCY", "ID", "ORTH", "LOWER", "NORM", "SHAPE", "PREFIX", "SUFFIX", "LENGTH",
    "LEMMA", "POS", "TAG", "DEP", "ENT_IOB", "ENT_TYPE", "ENT_ID", "ENT_KB_ID", "HEAD", "SENT_START", "SPACY", "LANG", "MORPH", "IDX",
]

# --- Helper Functions ---
# (get_done_ids, progress_piechart, force_sync_directory, save_document,
#  split_text_on_full_stop, log_nlp_statistics, log_system_metrics,
#  log_server_metrics, manage_memory functions remain unchanged)
# ... (paste unchanged helper functions here) ...
def get_done_ids(path: Path) -> List[str]:
    """Finds documents that have already been cleaned using pathlib."""
    ids = []
    if path.is_dir():
        ids = [p.stem for p in path.glob("*.spacy")]
    return ids

def progress_piechart(n_processed: int, n_total: int) -> go.Figure:
    """Draws piechart of progress"""
    # (Unchanged)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["done", "left"],
                values=[n_processed, max(0, n_total - n_processed)],
                textinfo="percent", hole=0.3,
                marker=dict(colors=["#36A2EB", "#FFCE56"]),
            )])
    fig.update_layout(
        title=f"Processing Progress: {n_processed}/{n_total} Documents",
        height=400, width=500)
    return fig

def force_sync_directory(directory_path: Path):
    """Force system to sync directory to disk (use RARELY)."""
    print("Warning: force_sync_directory called. This can severely impact performance.")
    try:
        if not directory_path.is_dir(): return
        if hasattr(os, 'sync'): os.sync()
        elif os.name == 'nt':
            import ctypes
            ctypes.windll.kernel32.FlushFileBuffers(ctypes.c_void_p(-1))
        if hasattr(os, 'fsync'):
            try:
                fd = os.open(str(directory_path), os.O_RDONLY)
                os.fsync(fd); os.close(fd)
            except OSError as e: print(f"Warning: Could not fsync directory {directory_path}: {e}")
            except Exception as e: print(f"Warning: General error fsyncing directory {directory_path}: {e}")
    except Exception as e: print(f"Warning: Could not force sync to disk: {e}")

def save_document(doc: Doc, dest_path: Path, config: RunConfig) -> Dict[str, Any]:
    """Serializes and saves spaCy Document using DocBin."""
    metrics = {"save_success": False, "save_time_sec": 0.0}
    start_time = time.time()
    try:
        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Use temporary file and rename for atomicity
        temp_dest = dest_path.with_suffix(dest_path.suffix + ".tmp")
        doc_bin = DocBin(attrs=TOKEN_ATTRS, docs=[doc])
        doc_bin.to_disk(str(temp_dest))
        temp_dest.rename(dest_path)
        metrics["save_success"] = True

        # Sync only if explicitly requested (HIGHLY DISCOURAGED)
        if config.force_sync:
            force_sync_directory(dest_path.parent)

    except Exception as e:
        error_msg = str(e)
        print(f"ERROR saving document {dest_path}: {error_msg}")
        if not config.disable_wandb and wandb.run:
            try:
                wandb.log({"document_save_errors": wandb.Table(
                    data=[[str(dest_path), error_msg]],
                    columns=["Path", "Error Message"]
                )})
            except Exception as log_e:
                print(f"Error logging save error to W&B: {log_e}")
    finally:
        metrics["save_time_sec"] = time.time() - start_time
    return metrics


def split_text_on_full_stop(text: str, max_length: int) -> list:
    """Splits text smartly respecting sentence boundaries."""
    # (Unchanged)
    segments = []; start = 0; text_length = len(text)
    while start < text_length:
        if text_length - start <= max_length:
            segments.append(text[start:].strip()); break
        slice_end = start + max_length
        segment = text[start:slice_end]
        split_index = segment.rfind('.')
        if split_index != -1: end = start + split_index + 1
        else:
            newline_index = segment.rfind('\n')
            if newline_index != -1: end = start + newline_index + 1
            else: end = slice_end
        segments.append(text[start:end].strip())
        start = end
    return [seg for seg in segments if seg]

def log_nlp_statistics(doc: Doc, step: int, doc_id: Optional[str], config: RunConfig):
    """Log NLP statistics for a document to wandb"""
    # (Unchanged)
    if config.disable_wandb or not wandb.run: return
    try:
        token_count = len(doc)
        if token_count == 0: wandb.log({"nlp_stats_warnings": f"Document {doc_id} has 0 tokens"}, step=step); return
        unique_tokens = len(set([token.text.lower() for token in doc]))
        pos_counts = {}; ent_counts = {}
        for token in doc: pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        for ent in doc.ents: ent_counts[ent.label_] = ent_counts.get(ent.label_, 0) + 1
        try: sentences = list(doc.sents); sentence_count = len(sentences)
        except: sentences=[]; sentence_count=0
        avg_sentence_length = token_count / sentence_count if sentence_count > 0 else 0
        token_lengths = [len(token.text) for token in doc]
        avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        has_sentiment = "sentiment" in doc.user_data
        sentiment = doc.user_data.get("sentiment", 0) if has_sentiment else None
        sentence_length_hist = None
        if sentence_count > 0:
            sentence_lengths = [len(sent) for sent in sentences]
            if sentence_lengths:
                 max_len = max(sentence_lengths, default=0)
                 sentence_length_bins = list(range(0, max_len + 10, 5))
                 if sentence_length_bins:
                     sentence_length_hist = wandb.Histogram(sentence_lengths, num_bins=min(20, len(sentence_length_bins)))
        stats_log = {"token_count": token_count, "unique_tokens": unique_tokens, "lexical_diversity": unique_tokens / token_count if token_count > 0 else 0,
                     "sentence_count": sentence_count, "avg_sentence_length": avg_sentence_length, "avg_token_length": avg_token_length, "entity_count": len(doc.ents)}
        if pos_counts: stats_log["pos_distribution"] = wandb.Table(data=[[pos, count, count/token_count] for pos, count in pos_counts.items()], columns=["POS", "Count", "Percentage"])
        if has_sentiment: stats_log["sentiment"] = sentiment
        if sentence_length_hist: stats_log["sentence_length_histogram"] = sentence_length_hist
        if ent_counts: stats_log["entity_distribution"] = wandb.Table(data=[[ent_type, count, count/len(doc.ents) if len(doc.ents) > 0 else 0] for ent_type, count in ent_counts.items()], columns=["Entity Type", "Count", "Percentage"])
        wandb.log(stats_log, step=step)
    except Exception as e:
        try: wandb.log({"nlp_stats_errors": wandb.Table(data=[[str(doc_id), str(e)]], columns=["Document ID", "Error"])}, step=step)
        except Exception as log_e: print(f"Error logging NLP stats error to W&B: {log_e}")

def log_system_metrics(config: RunConfig):
    """Log system resource usage metrics"""
    # (Unchanged)
    if config.disable_wandb or not wandb.run: return
    try:
        cpu_percent = psutil.cpu_percent(interval=None); cpu_times = psutil.cpu_times_percent(interval=None)
        memory = psutil.virtual_memory(); swap = psutil.swap_memory()
        try: disk = psutil.disk_usage(str(config.dest_dir))
        except FileNotFoundError: disk = None
        net_stats = {}; sys_load = {}
        try:
            net_io = psutil.net_io_counters()
            net_stats = {"net_bytes_sent": net_io.bytes_sent, "net_bytes_recv": net_io.bytes_recv, "net_packets_sent": net_io.packets_sent, "net_packets_recv": net_io.packets_recv}
        except Exception: pass
        try: load_avg = os.getloadavg(); sys_load = {"system_load_1min": load_avg[0], "system_load_5min": load_avg[1], "system_load_15min": load_avg[2]}
        except AttributeError: pass
        process = psutil.Process(); proc_mem = process.memory_info(); proc_cpu = process.cpu_percent(interval=None)
        log_payload = {"cpu_usage_percent": cpu_percent, "cpu_user_percent": cpu_times.user, "cpu_system_percent": cpu_times.system, "cpu_idle_percent": cpu_times.idle,
                       "memory_usage_percent": memory.percent, "memory_available_gb": memory.available / (1024 ** 3), "memory_used_gb": memory.used / (1024 ** 3), "memory_free_gb": memory.free / (1024 ** 3),
                       "swap_usage_percent": swap.percent if hasattr(swap, 'percent') else 0, "process_memory_rss_mb": proc_mem.rss / (1024 * 1024), "process_memory_vms_mb": proc_mem.vms / (1024 * 1024),
                       "process_cpu_percent": proc_cpu, "process_threads": process.num_threads(), **sys_load, **net_stats}
        if disk: log_payload.update({"disk_usage_percent": disk.percent, "disk_free_gb": disk.free / (1024 ** 3)})
        wandb.log(log_payload)
    except Exception as e:
         try: wandb.log({"system_metrics_error": str(e)})
         except Exception as log_e: print(f"Error logging system metrics error to W&B: {log_e}")

def log_server_metrics(config: RunConfig):
    """Enhanced logging for server environments"""
    # (Unchanged)
    if config.disable_wandb or not wandb.run or not numa: return
    try:
        server_log = {}; numa_stats = {}
        num_nodes = config.system_info.numa_nodes
        for node in range(num_nodes):
            try:
                free_mem, total_mem = numa.node_size(node)
                numa_stats[f"numa_node_{node}_free_memory_gb"] = free_mem / (1024**3)
                numa_stats[f"numa_node_{node}_total_memory_gb"] = total_mem / (1024**3)
                if total_mem > 0: numa_stats[f"numa_node_{node}_free_memory_percent"] = (free_mem / total_mem) * 100
            except Exception as numa_e: print(f"Warning: Could not get NUMA stats for node {node}: {numa_e}")
        if numa_stats: server_log["numa_stats"] = numa_stats
        try:
            per_cpu_percent = psutil.cpu_percent(percpu=True, interval=None)
            core_usage = {f"core_{i}": percent for i, percent in enumerate(per_cpu_percent)}
            if core_usage: server_log["core_usage"] = core_usage
        except Exception as cpu_e: print(f"Warning: Could not get per-core CPU usage: {cpu_e}")
        if server_log: wandb.log({"server_metrics": server_log})
    except Exception as e:
        print(f"Error logging server metrics: {e}")
        try: wandb.log({"server_metrics_error": str(e)})
        except Exception as log_e: print(f"Error logging server metrics error to W&B: {log_e}")

def manage_memory(step: int):
    """Periodically clean up memory"""
    # (Unchanged)
    try:
        collected = gc.collect(); print(f"\n[{step}] Garbage Collector: Freed {collected} objects.")
        if torch.cuda.is_available(): torch.cuda.empty_cache(); print(f"[{step}] Cleared CUDA cache.")
    except Exception as e: print(f"Error during memory management at step {step}: {e}")

# --- Core Processing Function (with DEBUG prints) ---

def process_document(text: str, doc_id: str, dest_path: Path, nlp: Language, config: RunConfig) -> Tuple[bool, Dict[str, Any]]:
    """
    Processes a single document text, handles splitting, uses nlp.pipe, saves result.
    Includes DEBUG print statements for single-file testing.

    Returns:
        Tuple (success_status, metrics_dict)
    """
    # <<< DEBUG PRINT ADDED >>>
    print(f"\n[DEBUG {doc_id}] Entering process_document. Text length: {len(text)}")
    process_start_time = time.time()
    metrics = {
        "doc_id": doc_id, "doc_length": len(text), "had_to_split": False, "num_segments": 1,
        "total_processing_time_sec": 0.0, "split_time_sec": 0.0, "pipe_time_sec": 0.0,
        "combine_time_sec": 0.0, "save_time_sec": 0.0, "token_count": 0, "sentence_count": 0,
        "entity_count": 0, "error_message": None,
    }
    success = False
    final_doc: Optional[Doc] = None

    try:
        # 1. Splitting (if necessary)
        split_start_time = time.time()
        # <<< DEBUG PRINT ADDED >>>
        print(f"[DEBUG {doc_id}] Checking if splitting needed (max_length={config.max_length})...")
        if len(text) > config.max_length:
            metrics["had_to_split"] = True
            segment_size = config.max_length
            texts = split_text_on_full_stop(text, int(segment_size))
            metrics["num_segments"] = len(texts)
            metrics["segment_size_used"] = segment_size
            # <<< DEBUG PRINT ADDED >>>
            print(f"[DEBUG {doc_id}] Splitting done. Segments: {metrics['num_segments']}")
            if not texts:
                 print(f"Warning: Splitting doc {doc_id} resulted in zero segments.")
                 texts = []
        else:
            texts = [text]
             # <<< DEBUG PRINT ADDED >>>
            print(f"[DEBUG {doc_id}] No splitting needed.")
        metrics["split_time_sec"] = time.time() - split_start_time

        # 2. spaCy Processing (using nlp.pipe)
        pipe_start_time = time.time()
        docs: List[Doc] = []
        if texts:
            non_empty_texts = [t for t in texts if t]
            if non_empty_texts:
                 # Determine n_process used based on whether splitting happened
                 # If not split, use n_process=1 regardless of config to avoid overhead for single item
                 current_n_process = config.n_process if metrics["had_to_split"] else 1
                 current_batch_size = config.batch_size # Keep configured batch_size? Or adjust if n_process=1? Let's keep it simple.
                 # <<< DEBUG PRINT ADDED >>>
                 print(f"[DEBUG {doc_id}] Calling nlp.pipe with n_process={current_n_process}, batch_size={current_batch_size} for {len(non_empty_texts)} segments...")
                 docs = list(nlp.pipe(non_empty_texts,
                                      n_process=current_n_process,
                                      batch_size=current_batch_size))
                 # <<< DEBUG PRINT ADDED >>>
                 print(f"[DEBUG {doc_id}] nlp.pipe finished.")
            else:
                 print(f"Warning: No non-empty text segments found for doc {doc_id} after splitting.")
        metrics["pipe_time_sec"] = time.time() - pipe_start_time

        # 3. Combine Docs (if split)
        # <<< DEBUG PRINT ADDED >>>
        print(f"[DEBUG {doc_id}] Combining {len(docs)} processed docs...")
        combine_start_time = time.time()
        if len(docs) > 1:
            final_doc = Doc.from_docs(docs)
        elif len(docs) == 1:
            final_doc = docs[0]
        else:
            final_doc = Doc(nlp.vocab)
        # <<< DEBUG PRINT ADDED >>>
        print(f"[DEBUG {doc_id}] Combining finished.")
        metrics["combine_time_sec"] = time.time() - combine_start_time

        # 4. Get Basic Doc Stats
        metrics["token_count"] = len(final_doc)
        metrics["entity_count"] = len(final_doc.ents)
        try:
            metrics["sentence_count"] = len(list(final_doc.sents)) if len(final_doc) > 0 else 0
        except Exception:
            metrics["sentence_count"] = 0

        # 5. Save Document
        # <<< DEBUG PRINT ADDED >>>
        print(f"[DEBUG {doc_id}] Saving document...")
        save_metrics = save_document(final_doc, dest_path, config)
        metrics.update(save_metrics)
        # <<< DEBUG PRINT ADDED >>>
        print(f"[DEBUG {doc_id}] Saving finished. Success: {metrics['save_success']}")

        success = metrics["save_success"]

    except Exception as e:
        # (Error handling unchanged)
        error_msg = f"Error processing doc {doc_id}: {str(e)}"
        print(f"\nCRITICAL during processing: {error_msg}")
        metrics["error_message"] = str(e)
        success = False
        if not config.disable_wandb and wandb.run:
             try: wandb.log({"processing_errors": wandb.Table(data=[[doc_id, str(dest_path), str(e)]], columns=["Document ID", "Document Path", "Error Message"])})
             except Exception as log_e: print(f"Error logging processing error: {log_e}")

    finally:
        metrics["total_processing_time_sec"] = time.time() - process_start_time
        # <<< DEBUG PRINT ADDED >>>
        print(f"[DEBUG {doc_id}] Exiting process_document. Total time: {metrics['total_processing_time_sec']:.2f}s")

    # Return final_doc object along with success and metrics for potential use in logging NLP stats
    return success, metrics, final_doc


# --- CLI Parser ---

def create_parser() -> argparse.ArgumentParser:
    """Creates parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="Corpus processor",
        description="Processes documents using spaCy (single model load, nlp.pipe parallelism).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Paths
    parser.add_argument("--src_index", type=str, required=True, help="Path to the CSV index file.")
    parser.add_argument("--dest_dir", type=str, required=True, help="Destination directory for .spacy files.")
    # Model
    parser.add_argument("--model_name", type=str, default="en_core_web_trf", help="Name of the spaCy model.")
    # Performance Tuning
    parser.add_argument("--max_length", type=int, default=0, help="Max doc length (chars) before splitting. Default: Auto.")
    parser.add_argument("--n_process", type=int, default=0, help="Processes for nlp.pipe. Default: Auto. TUNABLE.")
    parser.add_argument("--batch_size", type=int, default=0, help="Batch size for nlp.pipe. Default: Auto. TUNABLE.")
    parser.add_argument("--torch_threads", type=int, default=0, help="Max PyTorch threads. Default: Auto.")
    # Execution Control
    parser.add_argument("--force_sync", action="store_true", help="Force disk sync after saves (VERY SLOW).")
    parser.add_argument("--sample_interval", type=int, default=100, help="Log detailed metrics every N docs.")
    parser.add_argument("--checkpoint_interval", type=int, default=500, help="Save checkpoint every N docs.")
    # --- Test Mode Arguments ---
    parser.add_argument("--test_doc_id", type=str, default=None, help="If set, process only this document ID and exit.")
    parser.add_argument("--test_src_path", type=str, default=None, help="Optional: Direct path to source text file for test mode.")
    # --------------------------
    # W&B
    parser.add_argument("--wandb_user", type=str, default=os.getenv("WANDB_ENTITY"), help="W&B user/entity.")
    parser.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT", "corpus-processing"), help="W&B project.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional W&B run name.")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B logging.")

    return parser

# --- Main Execution ---

def main():
    """Main function to orchestrate corpus processing."""
    parser = create_parser()
    args = parser.parse_args()

    # --- Initialize Configuration ---
    try:
        config = RunConfig(**vars(args))
    except Exception as config_e:
        print(f"Error initializing configuration: {config_e}")
        sys.exit(1)

    print( # (print header unchanged)
        "--------------------------------------\n"
        f"-- PROCESS CORPUS (Run: {config.run_name}) --\n"
        "--- Single Model Load / nlp.pipe Parallelism ---\n"
        "--------------------------------------\n" )
    # (print config unchanged)
    print(f"Using Configuration:")
    print(f"  Model: {config.model_name}"); print(f"  Source Index: {config.src_index}"); print(f"  Destination: {config.dest_dir}")
    print(f"  Max Length: {config.max_length}"); print(f"  Pipe Processes: {config.n_process}"); print(f"  Pipe Batch Size: {config.batch_size}")
    print(f"  Torch Threads: {config.torch_threads}"); print(f"  W&B Disabled: {config.disable_wandb}")
    if config.test_doc_id: print(f"  TEST MODE ACTIVE for Doc ID: {config.test_doc_id}")
    print("--------------------------------------\n")

    # --- Directory Setup ---
    config.dest_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_dir = config.dest_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # --- Torch Thread Setup (Global) ---
    print(f"Setting global PyTorch threads to: {config.torch_threads}")
    torch.set_num_threads(config.torch_threads)

    # --- Initialize W&B ---
    # (W&B init unchanged)
    if not config.disable_wandb:
        if not config.wandb_user or not config.wandb_project:
             print("Warning: WANDB_ENTITY/--wandb_user or WANDB_PROJECT/--wandb_project not set. Disabling W&B.")
             config.disable_wandb = True
        else:
            print(f"Initializing W&B (User: {config.wandb_user}, Project: {config.wandb_project}, Run: {config.run_name})...")
            try:
                wandb.init(project=config.wandb_project, entity=config.wandb_user, name=config.run_name,
                           tags=["corpus-processing", config.model_name, "single-model"])
                config.log_to_wandb()
            except Exception as wandb_e:
                 print(f"Error initializing WandB: {wandb_e}. Disabling W&B for this run.")
                 config.disable_wandb = True
    else: print("Weights & Biases logging explicitly disabled.")


    # --- NUMA Setup ---
    # (NUMA setup unchanged)
    numa_enabled_runtime = False
    if config.system_info.numa_nodes > 1 and numa:
        numa_enabled_runtime = set_numa_awareness()
        print(f"NUMA awareness runtime status: {'Enabled' if numa_enabled_runtime else 'Not available/active'}")
        if not config.disable_wandb and wandb.run: wandb.config.update({"numa_enabled_runtime": numa_enabled_runtime})


    # --- Load spaCy Model ---
    # (Model loading unchanged)
    print(f"\nLoading spaCy model: {config.model_name}...")
    nlp: Optional[Language] = None; start_time_model_load = time.time()
    try:
        with warnings.catch_warnings():
             warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
             nlp = spacy.load(config.model_name)
        nlp.max_length = config.max_length
        model_load_time = time.time() - start_time_model_load
        mem_after_load = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"Model loaded successfully in {model_load_time:.2f} seconds."); print(f"Memory after model load: {mem_after_load:.2f} MB")
        if not config.disable_wandb and wandb.run:
            wandb.log({"perf/model_load_time_sec": model_load_time, "mem/after_load_mb": mem_after_load})
            try:
                pipeline_components = [{"name": name, "type": str(type(component))} for name, component in nlp.pipeline]
                wandb.log({"model/pipeline": wandb.Table(data=[[comp["name"], comp["type"]] for comp in pipeline_components], columns=["Component", "Type"])})
            except Exception as pipe_e: print(f"Warning: Could not log pipeline components: {pipe_e}")
    except Exception as e:
        error_message = f"Failed to load model '{config.model_name}': {str(e)}"
        print(f"CRITICAL ERROR: {error_message}")
        if not config.disable_wandb and wandb.run: wandb.log({"critical_errors": wandb.Table(data=[["Model Loading", error_message]], columns=["Stage", "Error"])})
        if wandb.run: wandb.finish(exit_code=1); sys.exit(1)


    # --- Load Index (Needed even for test mode to find path unless provided) ---
    print(f"\nLoading index: {config.src_index}")
    parsed_index: Optional[pd.DataFrame] = None
    try:
        parsed_index = pd.read_csv(config.src_index)
        required_cols = ['document_id', 'dest_path']
        if not all(col in parsed_index.columns for col in required_cols):
             raise ValueError(f"Index CSV must contain columns: {required_cols}")
        parsed_index['document_id'] = parsed_index['document_id'].astype(str)
        n_total_in_index = len(parsed_index.index)
        print(f"Loaded index with {n_total_in_index} documents.")
        if not config.disable_wandb and wandb.run:
            wandb.log({"data/total_documents_in_index": n_total_in_index})
    except Exception as e:
        # Don't exit if in test mode and path is provided, otherwise critical
        if not config.test_doc_id or not config.test_src_path:
            error_message = f"Failed to load or validate index '{config.src_index}': {str(e)}"
            print(f"CRITICAL ERROR: {error_message}")
            if not config.disable_wandb and wandb.run:
                 wandb.log({"critical_errors": wandb.Table(data=[["Index Loading", error_message]], columns=["Stage", "Error"])})
                 wandb.finish(exit_code=1)
            sys.exit(1)
        else:
            print(f"Warning: Failed to load index, but continuing in test mode with provided --test_src_path: {e}")
            parsed_index = None # Ensure it's None


    # ==============================================================
    # ======= SINGLE FILE TEST BLOCK ===============================
    # ==============================================================
    if config.test_doc_id:
        print("\n*** RUNNING SINGLE FILE TEST ***")
        TEST_DOC_ID = config.test_doc_id
        TEST_SRC_PATH = config.test_src_path # Already a Path object or None

        # Find source path from index if not provided directly
        if not TEST_SRC_PATH and parsed_index is not None:
            print(f"  Looking up source path for ID '{TEST_DOC_ID}' in index...")
            doc_row = parsed_index[parsed_index['document_id'] == TEST_DOC_ID]
            if doc_row.empty:
                print(f"ERROR: Test Document ID '{TEST_DOC_ID}' not found in index '{config.src_index}'.")
                print("Provide the path directly using --test_src_path or choose a valid ID.")
                if wandb.run: wandb.finish(exit_code=1)
                sys.exit(1)
            TEST_SRC_PATH = Path(doc_row.iloc[0]['dest_path']) # Get path from index
            print(f"  Found source path from index: {TEST_SRC_PATH}")
        elif not TEST_SRC_PATH:
             print(f"ERROR: Index could not be loaded and --test_src_path was not provided.")
             if wandb.run: wandb.finish(exit_code=1)
             sys.exit(1)

        TEST_DEST_PATH = config.dest_dir / f"{TEST_DOC_ID}.spacy"

        print(f"  Test Doc ID: {TEST_DOC_ID}")
        print(f"  Source Path: {TEST_SRC_PATH}")
        print(f"  Dest Path: {TEST_DEST_PATH}")
        print(f"  Using n_process: {config.n_process}")
        print(f"  Using batch_size: {config.batch_size}")
        print(f"  Using max_length: {config.max_length}")

        # Read the test file
        try:
            if not TEST_SRC_PATH.is_file():
                print(f"ERROR: Test source file not found: {TEST_SRC_PATH}")
                sys.exit(1)
            with open(TEST_SRC_PATH, 'r', encoding='utf-8') as f:
                test_text = f.read()
            print(f"  Read {len(test_text)} characters from test file.")
            if len(test_text) <= config.max_length:
                 print("  WARNING: Test file is NOT longer than max_length. nlp.pipe multiprocessing might not use multiple processes.")

        except Exception as e:
            print(f"ERROR reading test file {TEST_SRC_PATH}: {e}")
            sys.exit(1)

        # Call process_document for the test file
        print(f"\n[TEST] Starting processing for {TEST_DOC_ID}...")
        # The third element returned is final_doc, useful for debugging NLP stats if needed
        test_success, test_metrics, _ = process_document(
            test_text,
            doc_id=TEST_DOC_ID,
            dest_path=TEST_DEST_PATH,
            nlp=nlp,
            config=config
        )

        # Print Results
        print("\n*** SINGLE FILE TEST COMPLETE ***")
        print(f"  Success: {test_success}")
        print(f"  Metrics:")
        for key, value in test_metrics.items():
            if isinstance(value, float):
                 print(f"    {key}: {value:.4f}")
            else:
                 print(f"    {key}: {value}")

        # Exit cleanly
        print("\nExiting after single file test.")
        if not config.disable_wandb and wandb.run:
            wandb.log({"test_mode_results": test_metrics}) # Log test metrics
            wandb.finish(exit_code=0)
        sys.exit(0)
    # ==============================================================
    # ===== END OF SINGLE FILE TEST BLOCK ==========================
    # ==============================================================


    # --- Filter Processed Files (Full Run) ---
    print(f"\nChecking for already processed files in {config.dest_dir}...")
    done_ids = get_done_ids(config.dest_dir)
    n_done = len(done_ids)
    parsed_index_filtered = parsed_index # Start with the full loaded index
    if n_done > 0:
        done_filter = parsed_index['document_id'].isin(done_ids)
        n_already_done_in_index = done_filter.sum()
        print(f"Found {n_done} existing .spacy files. Ignoring {n_already_done_in_index} matching entries in index.")
        parsed_index_filtered = parsed_index[~done_filter].copy()
    else:
        print("No previously processed files found.")

    n_left = len(parsed_index_filtered.index)
    if n_left == 0:
        print("\nNo documents left to process.")
        if not config.disable_wandb and wandb.run: wandb.finish()
        sys.exit(0)

    print(f"Documents to process in this run: {n_left}")
    if not config.disable_wandb and wandb.run:
        wandb.log({"data/documents_to_process": n_left,
                   "data/documents_already_processed": n_done,
                   "progress/initial_pie": Plotly(progress_piechart(n_done, n_left + n_done))})


    # --- Prepare Iteration Data (Full Run) ---
    print("\nPreparing data for iteration...")
    try:
        src_paths = parsed_index_filtered['dest_path'].tolist()
        doc_ids = parsed_index_filtered['document_id'].tolist()
        doc_filenames = [config.dest_dir / f"{doc_id}.spacy" for doc_id in doc_ids]
        texts_stream = stream_files(src_paths)
        print("Data prepared.")
    except KeyError as e:
        print(f"CRITICAL ERROR: Missing column in index file used for iteration: {e}"); sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR preparing iteration data: {e}"); sys.exit(1)


    # --- Initialize Tracking (Full Run) ---
    start_time_total = time.time()
    processed_chars_total = 0
    successful_docs = 0
    failed_docs = 0
    total_processed_in_loop = 0
    all_metrics: List[Dict] = [] # Store metrics


    # --- Main Processing Loop (Full Run) ---
    print(f"\nStarting processing {n_left} documents...")
    pbar = tqdm(zip(doc_filenames, texts_stream, doc_ids), total=n_left, desc="Processing Docs", unit="doc")
    last_saved_doc : Optional[Doc] = None # Keep track of last successful Doc for NLP stats

    for doc_out_path, text, doc_id in pbar:
        print(f"\n---> Processing Document ID: {doc_id} ({total_processed_in_loop + 1}/{n_left})") # Live feedback

        # Process the document
        success, metrics, final_doc = process_document( # Get final_doc back
            text, doc_id=doc_id, dest_path=doc_out_path, nlp=nlp, config=config
        )
        all_metrics.append(metrics)
        if success and final_doc:
            last_saved_doc = final_doc # Store for potential NLP logging

        # --- Update Counters ---
        # (Counter logic unchanged)
        total_processed_in_loop += 1; doc_length = metrics.get("doc_length", 0)
        if doc_length > 0: processed_chars_total += doc_length
        if success: successful_docs += 1
        else: failed_docs += 1


        # --- Live Logging ---
        # (Live logging logic unchanged)
        if not config.disable_wandb and wandb.run:
            current_doc_global_index = n_done + total_processed_in_loop; elapsed_time = time.time() - start_time_total
            docs_per_sec = total_processed_in_loop / elapsed_time if elapsed_time > 0 else 0
            chars_per_sec = processed_chars_total / elapsed_time if elapsed_time > 0 else 0
            est_rem_time_sec = (elapsed_time / total_processed_in_loop) * (n_left - total_processed_in_loop) if total_processed_in_loop > 0 else 0
            log_data = {
                "progress/n_processed_total": current_doc_global_index, "progress/n_processed_this_run": total_processed_in_loop,
                "progress/percent": (current_doc_global_index / (n_left + n_done)) * 100 if (n_left + n_done) > 0 else 0,
                "progress/current_doc_id": doc_id, "perf/doc_length": metrics.get("doc_length", 0),
                "perf/total_processing_time_sec": metrics.get("total_processing_time_sec", 0.0), "perf/avg_docs_per_second": docs_per_sec,
                "perf/avg_chars_per_second": chars_per_sec, "time/elapsed_minutes": elapsed_time / 60,
                "time/estimated_remaining_minutes": est_rem_time_sec / 60, "stats/cumulative_successful_docs": successful_docs,
                "stats/cumulative_failed_docs": failed_docs, "perf/split_time_sec": metrics.get("split_time_sec", 0.0),
                "perf/pipe_time_sec": metrics.get("pipe_time_sec", 0.0), "perf/combine_time_sec": metrics.get("combine_time_sec", 0.0),
                "perf/save_time_sec": metrics.get("save_time_sec", 0.0), }
            try: wandb.log(log_data)
            except Exception as log_e: print(f"Error logging W&B data: {log_e}")

        # --- Periodic Tasks ---
        if total_processed_in_loop % config.checkpoint_interval == 0 and total_processed_in_loop > 0:
            # (Checkpoint logic unchanged)
            checkpoint_path = checkpoint_dir / f"checkpoint_{n_done + total_processed_in_loop}.csv"
            try:
                chkpt_df = pd.DataFrame(all_metrics); chkpt_df['timestamp'] = time.time()
                chkpt_df.to_csv(checkpoint_path, index=False)
                print(f"\nSaved checkpoint ({len(chkpt_df)} rows) to {checkpoint_path}")
                if not config.disable_wandb and wandb.run:
                    try:
                         chkpt_artifact = wandb.Artifact(f"checkpoint_{n_done + total_processed_in_loop}", type="checkpoint")
                         chkpt_artifact.add_file(str(checkpoint_path)); wandb.log_artifact(chkpt_artifact)
                    except Exception as log_e: print(f"Error logging checkpoint artifact: {log_e}")
            except Exception as cp_e: print(f"\nError saving checkpoint: {cp_e}")

        if total_processed_in_loop % config.sample_interval == 0 and total_processed_in_loop > 0:
            print(f"\n[{total_processed_in_loop}] Logging detailed metrics...")
            log_system_metrics(config); log_server_metrics(config)
            # Log NLP stats using the *last successfully saved Doc object*
            if last_saved_doc:
                 try:
                      current_global_step = n_done + total_processed_in_loop
                      print(f"[{total_processed_in_loop}] Logging NLP stats for last successful doc ({last_saved_doc._.get('doc_id','N/A')})...") # Assumes you set doc_id in user_data if needed
                      log_nlp_statistics(last_saved_doc, current_global_step, last_saved_doc._.get('doc_id','N/A'), config)
                 except Exception as e:
                      print(f"\nError logging NLP stats: {e}")
                      if not config.disable_wandb and wandb.run: wandb.log({"statistics_errors": str(e)})
            else:
                 print(f"[{total_processed_in_loop}] Skipping NLP stats (no successful doc saved recently).")

        if total_processed_in_loop % (config.checkpoint_interval * 2) == 0 and total_processed_in_loop > 0:
            manage_memory(total_processed_in_loop)

        pbar.set_description(f"Last: {doc_id[:15]} ({metrics.get('total_processing_time_sec', 0.0):.2f}s)")

    # --- End of Loop ---
    pbar.close()
    print("\nProcessing loop finished.")


    # --- Final Index and Checkpoint ---
    # (Final save logic unchanged)
    print("\nCreating final index and saving final checkpoint...")
    try:
        final_df = pd.DataFrame(all_metrics)
        final_df['processing_status'] = final_df['error_message'].apply(lambda x: 'failed' if pd.notna(x) else 'success')
        if parsed_index is not None: # Only map if index was loaded
             id_to_src_map = pd.Series(parsed_index['dest_path'].values, index=parsed_index['document_id']).to_dict()
             final_df['src_path'] = final_df['doc_id'].map(id_to_src_map)
        else: # Add placeholder if index failed
             final_df['src_path'] = 'N/A (Index Load Failed)'
        final_df['dest_path'] = final_df['doc_id'].map(lambda did: str(config.dest_dir / f"{did}.spacy"))
        index_cols = ['doc_id', 'src_path', 'dest_path', 'processing_status']
        final_index = final_df.reindex(columns=index_cols, fill_value=None) # Ensure columns exist
        index_path = config.dest_dir / f"index_{config._run_id}.csv"
        final_index.to_csv(index_path, index=False)
        print(f"Saved final index ({len(final_index)} entries) to {index_path}")
        if not config.disable_wandb and wandb.run:
             try:
                  index_artifact = wandb.Artifact(f"processed_index_{config.run_name}", type="dataset_index")
                  index_artifact.add_file(str(index_path)); wandb.log_artifact(index_artifact)
             except Exception as log_e: print(f"Error logging index artifact: {log_e}")
        final_checkpoint_path = checkpoint_dir / f"final_metrics_{config._run_id}.csv"
        final_df.to_csv(final_checkpoint_path, index=False)
        print(f"Saved final metrics checkpoint ({len(final_df)} rows) to {final_checkpoint_path}")
        if not config.disable_wandb and wandb.run:
            try:
                metrics_artifact = wandb.Artifact(f"final_metrics_{config.run_name}", type="run_metrics")
                metrics_artifact.add_file(str(final_checkpoint_path)); wandb.log_artifact(metrics_artifact)
            except Exception as log_e: print(f"Error logging metrics artifact: {log_e}")
    except Exception as final_e: print(f"Error during final index/checkpoint creation: {final_e}")


    # --- Final Summary ---
    # (Summary logic unchanged)
    print("\nCalculating final summary...")
    processing_time_total_sec = time.time() - start_time_total; actual_processed_count = total_processed_in_loop
    success_rate_float = (successful_docs / actual_processed_count * 100) if actual_processed_count > 0 else 0.0
    avg_proc_time_sec = (processing_time_total_sec / actual_processed_count) if actual_processed_count > 0 else 0.0
    avg_chars_per_sec = (processed_chars_total / processing_time_total_sec) if processing_time_total_sec > 0 else 0.0
    summary_data = [
        ("Total Documents in Index", n_total_in_index if parsed_index is not None else "N/A"),
        ("Documents Already Processed (Skipped)", n_done), ("Documents Attempted in This Run", actual_processed_count),
        ("Successfully Processed (this run)", successful_docs), ("Failed (this run)", failed_docs),
        ("Success Rate (this run)", success_rate_float), ("Total Processing Time (min)", processing_time_total_sec / 60),
        ("Avg Time per Document (sec, this run)", avg_proc_time_sec), ("Avg Processing Speed (chars/sec, this run)", avg_chars_per_sec),
        ("Total Characters Processed (this run)", processed_chars_total), ("Config: spaCy Processes (nlp.pipe)", config.n_process),
        ("Config: spaCy Batch Size (nlp.pipe)", config.batch_size), ("Config: Max Document Length (chars)", config.max_length),
        ("Config: PyTorch Threads", config.torch_threads), ]
    print("\n===== Processing Summary =====")
    for name, value in summary_data:
        if name == "Success Rate (this run)": print(f"{name}: {value:.2f}%")
        elif isinstance(value, float): print(f"{name}: {value:.2f}")
        else: print(f"{name}: {value}")
    if not config.disable_wandb and wandb.run:
        print("\nLogging final summary to W&B...")
        try:
            summary_table_data = [[item[0], item[1]] for item in summary_data]
            wandb.log({"final_summary": wandb.Table(data=summary_table_data, columns=["Metric", "Value"])})
            wandb.log({ "perf/avg_doc_time_sec": avg_proc_time_sec, "perf/avg_chars_per_sec": avg_chars_per_sec,
                "stats/success_rate_percent": success_rate_float, "time/total_runtime_min": processing_time_total_sec / 60 })
            print("Finishing W&B run..."); wandb.finish(); print("W&B run finished.")
        except Exception as log_e: print(f"Error logging final summary / finishing WandB: {log_e}")

    print("\nScript finished.")


if __name__ == "__main__":
    main()
