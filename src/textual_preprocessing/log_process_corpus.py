# -*- coding: utf-8 -*-
"""Script responsible for cleaning/processing the corpus."""
import argparse
import gc # For memory management
import glob
# import multiprocessing # concurrent.futures is used instead for main loop
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed # For batch processing

import pandas as pd
import plotly.graph_objects as go
import psutil
import spacy
import torch
import wandb
from spacy.language import Language
from spacy.tokens import Doc, DocBin
from tqdm import tqdm
# Assuming utils.streams.stream_files reads files lazily - not used in batch mode
# from utils.streams import stream_files
from wandb.data_types import Plotly

      
# Try importing NUMA libraries
try:
    import numexpr
except ImportError:
    numexpr = None

try:
    import numa
    # Perform a basic check by trying to access a core function.
    # If this works without error, we assume the library is functional
    # and NUMA is likely supported to some extent by the system.
    _ = numa.num_configured_nodes() # Example check
    print("NUMA library ('pynuma') imported successfully.")
except ImportError:
    print("NUMA Python library (e.g., 'pynuma') not installed. NUMA awareness disabled.")
    print("Suggestion: pip install pynuma")
    numa = None
except AttributeError:
    # Handle cases where the library might exist but lack expected functions (less likely with pynuma)
     print("Imported 'numa' library seems incomplete or incompatible.")
     numa = None
except Exception as e:
    # Catch other errors during the initial check (e.g., syscall errors if NUMA not supported by OS/kernel)
    print(f"NUMA library loaded but failed initial check (NUMA likely not supported by OS/hardware): {e}")
    numa = None

# --- System Resource Configuration ---


def configure_system_resources():
    """Configure system resources optimized for high-core-count Xeon servers"""
    cpu_count = os.cpu_count()
    available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)

    # --- CRITICAL CHANGE HERE ---
    # N_PROCESS_PIPE: Set to 1. Eliminate nlp.pipe parallelism within workers.
    n_process_pipe = 1 # <<< MOST IMPORTANT CHANGE

    # BATCH_SIZE_PIPE: Keep reasonable, but maybe slightly smaller?
    batch_size_pipe = max(16, min(int(available_memory_gb * 2), 96)) # Slightly reduced max

    # MAX_LENGTH: Keep as before
    max_length = min(2 * 10**4, int(available_memory_gb * 8 * 10**3))
    max_length = max(10000, min(max_length, 2 * 10**6))

    # NUMA_NODES: Keep as before
    numa_nodes = 2
    if numa:
        try:
            num_nodes_detected = numa.num_configured_nodes()
            if num_nodes_detected > 0:
                numa_nodes = num_nodes_detected
            print(f"Detected {numa_nodes} NUMA nodes.")
        except Exception as e:
            print(f"Could not detect NUMA nodes via library, using default {numa_nodes}. Error: {e}")

    # TORCH_THREADS: Reduce significantly as inner parallelism is gone.
    # Base it on cores available *per worker* (total cores / numa nodes)
    # but don't go too high. Maybe 4-8 threads per worker.
    cores_per_worker = cpu_count // numa_nodes if numa_nodes > 0 else cpu_count
    torch_threads = max(2, min(8, cores_per_worker // 2)) # Reduced from 16

    print(f"System Config: CPU={cpu_count}, MemGB={available_memory_gb:.2f}, CoresPerPipe={n_process_pipe}, PipeBatch={batch_size_pipe}, MaxLen={max_length}, TorchThreads={torch_threads}, NUMANodes={numa_nodes}")

    # Ensure keys match the rest of the script expects
    return {
        "MAX_LENGTH": max_length,
        "N_PROCESS_PIPE": n_process_pipe,
        "BATCH_SIZE_PIPE": batch_size_pipe,
        "CPU_COUNT": cpu_count,
        "AVAILABLE_MEMORY_GB": available_memory_gb,
        "TORCH_THREADS": torch_threads,
        "NUMA_NODES": numa_nodes
    }

# Rest of the SYSTEM_CONFIG setup remains the same, N_PROCESS_PIPE will be 1
SYSTEM_CONFIG = configure_system_resources()
MAX_LENGTH = SYSTEM_CONFIG["MAX_LENGTH"]
N_PROCESS_PIPE = SYSTEM_CONFIG["N_PROCESS_PIPE"]   # Will be 1
BATCH_SIZE_PIPE = SYSTEM_CONFIG["BATCH_SIZE_PIPE"]


# --- NUMA Awareness Setup ---

def set_numa_awareness():
    """Set NUMA awareness if available"""
    # (Function unchanged from previous version - relies on global `numa` and `numexpr`)
    numa_available = False
    try:
        if numexpr:
            numexpr.set_num_threads(SYSTEM_CONFIG["TORCH_THREADS"])
            print(f"NumExpr using {SYSTEM_CONFIG['TORCH_THREADS']} threads.")

        if numa: # Check if library was successfully imported and available
            print("NUMA library available and supported.")
            numa_available = True
            # Optional: Set affinity/policy here if desired, e.g.:
            # numa.set_interleave_mask(numa.all_nodes_mask)
            # print("Set NUMA policy to interleave.")
        else:
             print("NUMA library not installed or not supported on this system.")
        return numa_available
    except Exception as e:
        print(f"Error setting NUMA awareness: {e}")
        return False

# --- Constants ---
# (TOKEN_ATTRS unchanged)
TOKEN_ATTRS = [ # Consider reducing this list if not all attributes are needed
    "IS_ALPHA", "IS_ASCII", "IS_DIGIT", "IS_LOWER", "IS_PUNCT", "IS_SPACE",
    "IS_TITLE", "IS_UPPER", "LIKE_URL", "LIKE_NUM", "LIKE_EMAIL", "IS_STOP",
    "IS_QUOTE", "IS_BRACKET", "IS_LEFT_PUNCT", "IS_RIGHT_PUNCT", "IS_CURRENCY",
    "ID", "ORTH", "LOWER", "NORM", "SHAPE", "PREFIX", "SUFFIX", "LENGTH",
    "LEMMA", "POS", "TAG", "DEP", "ENT_IOB", "ENT_TYPE", "ENT_ID", "ENT_KB_ID",
    "HEAD", "SENT_START", "SPACY", "LANG", "MORPH", "IDX",
]


# --- Helper Functions ---
# (get_done_ids, progress_piechart, force_sync_directory, save_document,
#  split_text_on_full_stop, log_nlp_statistics, log_system_metrics,
#  log_server_metrics, manage_memory functions remain unchanged from previous version)
# ... (paste unchanged helper functions here) ...
def get_done_ids(path: str) -> List[str]:
    """Finds documents that have already been cleaned using pathlib"""
    dest_path = Path(path)
    ids = []
    if dest_path.is_dir():
        # Use stem to get filename without extension, handles ".spacy" correctly
        ids = [p.stem for p in dest_path.glob("*.spacy")]
    return ids

def progress_piechart(n_processed: int, n_total: int) -> go.Figure:
    """Draws piechart of progress"""
    # (Function unchanged)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["done", "left"],
                values=[n_processed, max(0, n_total - n_processed)], # Ensure 'left' isn't negative
                textinfo="percent",
                hole=0.3,
                marker=dict(colors=["#36A2EB", "#FFCE56"]),
            )
        ]
    )
    fig.update_layout(
        title=f"Processing Progress: {n_processed}/{n_total} Documents",
        height=400,
        width=500,
    )
    return fig

def force_sync_directory(directory_path):
    """Force system to sync directory to disk.
       NOTE: Calling this frequently (e.g., after every file) can
             significantly degrade performance. Consider removing or
             calling much less often.
    """
    # (Function unchanged)
    try:
        directory = Path(directory_path)
        if not directory.is_dir():
            return # Don't try to sync if dir doesn't exist

        if hasattr(os, 'sync'):  # Unix/Linux
            os.sync()
        elif os.name == 'nt':  # Windows
            import ctypes
            ctypes.windll.kernel32.FlushFileBuffers(ctypes.c_void_p(-1))

        # Additional platform-specific sync for directory descriptor
        if hasattr(os, 'fsync'):
            try:
                # Ensure directory exists before opening
                if directory.exists():
                    fd = os.open(str(directory), os.O_RDONLY)
                    os.fsync(fd)
                    os.close(fd)
            except OSError as e:
                # Ignore errors like "Is a directory" on some systems or permission errors
                 print(f"Warning: Could not fsync directory descriptor for {directory}: {e}")
            except Exception as e:
                 print(f"Warning: General error fsyncing directory {directory}: {e}")

    except Exception as e:
        print(f"Warning: Could not force sync to disk: {e}")

def save_document(doc: Doc, dest: str, disable_wandb: bool) -> None:
    """Serializes and saves spaCy Document."""
    start_time = time.time()
    success = True
    error_msg = ""
    dest_path = Path(dest)

    try:
        # Create directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Create and save DocBin
        doc_bin = DocBin(attrs=TOKEN_ATTRS, docs=[doc])
        # Use temporary file and rename for atomicity
        temp_dest = dest_path.with_suffix(dest_path.suffix + ".tmp")
        doc_bin.to_disk(str(temp_dest))
        temp_dest.rename(dest_path)

        # Force synchronization (Consider removing or reducing frequency)
        # force_sync_directory(dest_path.parent)

    except Exception as e:
        success = False
        error_msg = str(e)
        print(f"ERROR saving document {dest}: {error_msg}")
        if not disable_wandb:
            try:
                wandb.log({"document_save_errors": wandb.Table(
                    data=[[dest, error_msg]],
                    columns=["Path", "Error Message"]
                )})
            except Exception as log_e:
                print(f"Error logging save error to W&B: {log_e}")
        # Don't raise here, let the caller handle status
    finally:
        # Log save operation metrics
        save_time = time.time() - start_time
        if not disable_wandb:
            try:
                 wandb.log({
                    "document_save_time": save_time,
                    "document_save_success": success,
                    # "document_path": dest # Can be verbose
                 })
            except Exception as log_e:
                 print(f"Error logging save metrics to W&B: {log_e}")

def split_text_on_full_stop(text: str, max_length: int) -> list:
    """
    Splits the text into chunks of at most max_length characters,
    preferring to split at full stops. Falls back to newline, then hard split.
    """
    # (Function unchanged)
    segments = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # If the remaining text is short enough, append and break
        if text_length - start <= max_length:
            segments.append(text[start:].strip())
            break

        # Look for the last full stop in the allowed slice
        slice_end = start + max_length
        segment = text[start:slice_end]
        split_index = segment.rfind('.')

        if split_index != -1:
            # We found a full stop, include it in the segment
            end = start + split_index + 1
        else:
            # Fallback: try to break on newline
            newline_index = segment.rfind('\n')
            if newline_index != -1:
                end = start + newline_index + 1
            else:
                # No full stop or newline; split at max_length directly
                end = slice_end

        # Append the found segment and update start index
        segments.append(text[start:end].strip())
        start = end

    # Filter out empty strings that might result from consecutive delimiters
    return [seg for seg in segments if seg]

def log_nlp_statistics(doc, step, doc_id=None, disable_wandb=False):
    """Log NLP statistics for a document to wandb"""
    if disable_wandb: return
    # (Function largely unchanged, added disable_wandb check)
    try:
        token_count = len(doc)
        if token_count == 0:
            wandb.log({"nlp_stats_warnings": f"Document {doc_id} has 0 tokens"}, step=step)
            return

        # ... (rest of the calculations remain the same) ...
        unique_tokens = len(set([token.text.lower() for token in doc]))
        pos_counts = {}
        for token in doc: pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        ent_counts = {}
        for ent in doc.ents: ent_counts[ent.label_] = ent_counts.get(ent.label_, 0) + 1
        sentences = list(doc.sents)
        sentence_count = len(sentences)
        avg_sentence_length = token_count / sentence_count if sentence_count > 0 else 0
        token_lengths = [len(token.text) for token in doc]
        avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        has_sentiment = "sentiment" in doc.user_data
        sentiment = doc.user_data.get("sentiment", 0) if has_sentiment else None

        if sentence_count > 0:
            sentence_lengths = [len(sent) for sent in sentences]
            sentence_length_bins = list(range(0, max(sentence_lengths, default=0) + 10, 5))
            sentence_length_hist = wandb.Histogram(
                sentence_lengths,
                num_bins=min(20, len(sentence_length_bins))
            ) if sentence_length_bins else None # Handle case with no bins
        else:
            sentence_length_hist = None

        stats_log = {
            "token_count": token_count, "unique_tokens": unique_tokens,
            "lexical_diversity": unique_tokens / token_count if token_count > 0 else 0,
            "sentence_count": sentence_count, "avg_sentence_length": avg_sentence_length,
            "avg_token_length": avg_token_length, "entity_count": len(doc.ents),
        }
        if pos_counts:
            stats_log["pos_distribution"] = wandb.Table(
                data=[[pos, count, count/token_count] for pos, count in pos_counts.items()],
                columns=["POS", "Count", "Percentage"]
            )
        if has_sentiment: stats_log["sentiment"] = sentiment
        if sentence_length_hist: stats_log["sentence_length_histogram"] = sentence_length_hist
        if ent_counts:
             stats_log["entity_distribution"] = wandb.Table(
                data=[[ent_type, count, count/len(doc.ents) if len(doc.ents) > 0 else 0]
                      for ent_type, count in ent_counts.items()],
                columns=["Entity Type", "Count", "Percentage"]
            )

        wandb.log(stats_log, step=step)

    except Exception as e:
        try:
            wandb.log({"nlp_stats_errors": wandb.Table(
                data=[[str(doc_id), str(e)]],
                columns=["Document ID", "Error"]
            )}, step=step)
        except Exception as log_e:
            print(f"Error logging NLP stats error to W&B: {log_e}")

def log_system_metrics(disable_wandb=False):
    """Log system resource usage metrics"""
    if disable_wandb: return
    # (Function largely unchanged, added disable_wandb check)
    try:
        # ... (metric gathering remains the same) ...
        cpu_percent = psutil.cpu_percent(interval=None) # Use non-blocking call
        cpu_times = psutil.cpu_times_percent(interval=None)
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        try: disk = psutil.disk_usage('/')
        except FileNotFoundError: disk = None # Handle case where '/' might not be valid (e.g. Windows)
        net_stats = {}
        try:
            net_io = psutil.net_io_counters()
            net_stats = {"net_bytes_sent": net_io.bytes_sent, "net_bytes_recv": net_io.bytes_recv,
                         "net_packets_sent": net_io.packets_sent, "net_packets_recv": net_io.packets_recv}
        except Exception: pass
        try:
            load_avg = os.getloadavg()
            sys_load = {"system_load_1min": load_avg[0], "system_load_5min": load_avg[1], "system_load_15min": load_avg[2]}
        except AttributeError: sys_load = {} # Handle non-Unix systems

        process = psutil.Process()
        proc_mem = process.memory_info()
        proc_cpu = process.cpu_percent(interval=None) # Use non-blocking

        log_payload = {
            "cpu_usage_percent": cpu_percent, "cpu_user_percent": cpu_times.user,
            "cpu_system_percent": cpu_times.system, "cpu_idle_percent": cpu_times.idle,
            "memory_usage_percent": memory.percent, "memory_available_gb": memory.available / (1024 ** 3),
            "memory_used_gb": memory.used / (1024 ** 3), "memory_free_gb": memory.free / (1024 ** 3),
            "swap_usage_percent": swap.percent if hasattr(swap, 'percent') else 0,
            "process_memory_rss_mb": proc_mem.rss / (1024 * 1024),
            "process_memory_vms_mb": proc_mem.vms / (1024 * 1024),
            "process_cpu_percent": proc_cpu, "process_threads": process.num_threads(),
            **sys_load, **net_stats
        }
        if disk:
             log_payload.update({"disk_usage_percent": disk.percent, "disk_free_gb": disk.free / (1024 ** 3)})

        wandb.log(log_payload)
    except Exception as e:
         try:
             wandb.log({"system_metrics_error": str(e)})
         except Exception as log_e:
              print(f"Error logging system metrics error to W&B: {log_e}")

def log_server_metrics(disable_wandb=False):
    """Enhanced logging for server environments"""
    if disable_wandb or not numa: return # Skip if wandb disabled or numa not available/installed
    # (Function unchanged)
    try:
        server_log = {}
        # Log NUMA-specific metrics
        if numa: # We already checked for installation and availability at the start
            numa_stats = {}
            num_nodes = numa.num_configured_nodes()
            for node in range(num_nodes):
                try:
                    # node_size returns (free_size, total_size)
                    free_mem, total_mem = numa.node_size(node)
                    numa_stats[f"numa_node_{node}_free_memory_gb"] = free_mem / (1024**3)
                    numa_stats[f"numa_node_{node}_total_memory_gb"] = total_mem / (1024**3)
                    if total_mem > 0:
                        numa_stats[f"numa_node_{node}_free_memory_percent"] = (free_mem / total_mem) * 100
                except Exception as numa_e:
                     print(f"Warning: Could not get NUMA stats for node {node}: {numa_e}")
            if numa_stats:
                server_log["numa_stats"] = numa_stats

        # Log CPU topology info (per-core usage)
        try:
            per_cpu_percent = psutil.cpu_percent(percpu=True, interval=None) # Non-blocking
            core_usage = {f"core_{i}": percent for i, percent in enumerate(per_cpu_percent)}
            if core_usage:
                server_log["core_usage"] = core_usage
        except Exception as cpu_e:
            print(f"Warning: Could not get per-core CPU usage: {cpu_e}")

        if server_log: # Only log if we gathered something
            wandb.log({"server_metrics": server_log})

    except Exception as e:
        print(f"Error logging server metrics: {e}")
        try:
            wandb.log({"server_metrics_error": str(e)})
        except Exception as log_e:
             print(f"Error logging server metrics error to W&B: {log_e}")

def manage_memory():
    """Periodically clean up memory to prevent leaks on long-running jobs"""
    # (Function unchanged)
    try:
        collected = gc.collect()
        print(f"Garbage Collector: Freed {collected} objects.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA cache.")
    except Exception as e:
        print(f"Error during memory management: {e}")


# --- Document Processing Function ---

def process_document(text: str, nlp: Language, dest: str, doc_id: str = None, disable_wandb: bool = False) -> Tuple[bool, str, float, Dict]:
    """Turns text into a spaCy document. Uses nlp.pipe internally with
       configured N_PROCESS_PIPE and BATCH_SIZE_PIPE.

    Returns:
        Tuple containing (success status, error message if any, processing time, metrics dict)
    """
    # (Function mostly unchanged, but nlp.pipe call is the key difference)
    metrics = {
        "doc_length": len(text),
        "had_to_split": len(text) > MAX_LENGTH,
    }
    start_time = time.time()

    try:
        # Track memory usage within this specific document processing call if needed
        # peak_memory_before = psutil.Process().memory_info().rss / (1024 * 1024)

        segment_size = MAX_LENGTH # Default segment size
        if len(text) > MAX_LENGTH:
            # Splitting logic remains the same, based on MAX_LENGTH
            segment_size = MAX_LENGTH // 2 if SYSTEM_CONFIG["CPU_COUNT"] > 32 else MAX_LENGTH
            texts = split_text_on_full_stop(text, segment_size)
            metrics["num_segments"] = len(texts)
            metrics["avg_segment_length"] = sum(len(t) for t in texts) / len(texts) if texts else 0
            metrics["segment_size_used"] = segment_size

            if not disable_wandb:
                 try:
                     wandb.log({"text_splitting": wandb.Table(
                         data=[[doc_id, len(text), len(texts), MAX_LENGTH, segment_size]],
                         columns=["Document ID", "Original Length", "Number of Segments", "Max Length", "Segment Size Used"]
                     )})
                 except Exception as log_e: print(f"Error logging splitting table: {log_e}")
        else:
            texts = [text]
            metrics["num_segments"] = 1
            metrics["avg_segment_length"] = len(text)

        # --- CRITICAL CHANGE HERE ---
        # Use nlp.pipe() with the *reduced* N_PROCESS_PIPE and BATCH_SIZE_PIPE
        pipe_start_time = time.time()
        docs = list(nlp.pipe(texts, n_process=N_PROCESS_PIPE, batch_size=BATCH_SIZE_PIPE))
        pipe_time = time.time() - pipe_start_time
        metrics["pipe_processing_time"] = pipe_time
        # --------------------------

        # Combine the processed segments back into a single document
        # (Combine logic unchanged)
        combine_start_time = time.time()
        if len(docs) > 1:
            doc = Doc.from_docs(docs)
        elif len(docs) == 1:
            doc = docs[0]
        else:
            vocab = nlp.vocab
            doc = Doc(vocab)
            print(f"Warning: No segments processed for doc_id {doc_id}, creating empty Doc.")
        combine_time = time.time() - combine_start_time
        metrics["doc_combine_time"] = combine_time

        # Collect document statistics
        # (Stats collection unchanged)
        metrics["token_count"] = len(doc)
        metrics["entity_count"] = len(doc.ents)
        try:
            metrics["sentence_count"] = len(list(doc.sents)) if len(doc) > 0 else 0
        except Exception as sent_e:
            print(f"Warning: Could not count sentences for {doc_id}: {sent_e}")
            metrics["sentence_count"] = 0

        # Save document
        # (Save logic unchanged)
        save_start_time = time.time()
        save_document(doc, dest=dest, disable_wandb=disable_wandb)
        save_time = time.time() - save_start_time
        metrics["doc_save_time"] = save_time

        processing_time = time.time() - start_time
        metrics["total_processing_time"] = processing_time

        return True, "", processing_time, metrics

    except Exception as e:
        # (Error handling unchanged)
        error_message = f"Error in process_document for {doc_id}: {str(e)}"
        print(error_message)
        if not disable_wandb:
             try:
                 wandb.log({"processing_errors": wandb.Table(
                     data=[[doc_id, dest, str(e)]],
                     columns=["Document ID", "Document Path", "Error Message"]
                 )})
             except Exception as log_e: print(f"Error logging processing error: {log_e}")
        metrics["total_processing_time"] = time.time() - start_time
        return False, str(e), metrics.get("total_processing_time", 0.0), metrics


# --- Worker Function for ProcessPoolExecutor ---
# (process_batch function remains unchanged from previous version - it correctly passes SYSTEM_CONFIG)
def process_batch(batch_paths_ids_dests, nlp_model_name, system_config, disable_wandb):
    """Worker function to process a batch of documents."""
    worker_results = []
    worker_nlp = None
    pid = os.getpid()
    process = psutil.Process(pid) # Get process object for memory check

    mem_before_load = process.memory_info().rss / (1024 * 1024)
    print(f"[Worker {pid}] Memory before model load: {mem_before_load:.2f} MB")

    print(f"[Worker {pid}] Loading model '{nlp_model_name}'...")

    try:
        # Set PyTorch threads for this worker process using the passed config
        torch.set_num_threads(system_config["TORCH_THREADS"])
        print(f"[Worker {pid}] Set torch threads to {system_config['TORCH_THREADS']}")

        # Load model in each worker process
        worker_nlp = spacy.load(nlp_model_name)
        worker_nlp.max_length = system_config["MAX_LENGTH"] # Use max_length from config

        mem_after_load = process.memory_info().rss / (1024 * 1024)
        print(f"[Worker {pid}] Model loaded. Max length: {worker_nlp.max_length}. Memory after load: {mem_after_load:.2f} MB")

    except Exception as load_e:
        print(f"FATAL ERROR in worker {pid}: Could not load model {nlp_model_name}: {load_e}")
        error_msg = f"Worker {pid} failed to load model: {load_e}"
        for _, doc_id, _ in batch_paths_ids_dests:
             worker_results.append((doc_id, False, error_msg, 0.0, {"doc_length": 0}))
        return worker_results

    print(f"[Worker {pid}] Processing batch of {len(batch_paths_ids_dests)} documents...")
    mem_before_batch = process.memory_info().rss / (1024 * 1024)
    print(f"[Worker {pid}] Memory before processing batch: {mem_before_batch:.2f} MB")

    batch_start_time = time.time() # Time the batch processing itself

    for i, (src_path, doc_id, dest_path) in enumerate(batch_paths_ids_dests):
        text = None
        try:
            # Read the text file
            with open(src_path, 'r', encoding='utf-8') as f:
                text = f.read()
            if not text:
                 print(f"[Worker {pid}] Warning: Document {doc_id} ({src_path}) is empty.")

            # Process the document using the main function
            # It will use N_PROCESS_PIPE=1 and BATCH_SIZE_PIPE via globals
            success, error, proc_time, metrics = process_document(
                text if text is not None else "",
                worker_nlp,
                dest_path,
                doc_id,
                disable_wandb=disable_wandb
            )
            worker_results.append((doc_id, success, error, proc_time, metrics))

        except FileNotFoundError:
             error_msg = f"File not found: {src_path}"
             print(f"[Worker {pid}] ERROR: {error_msg}")
             worker_results.append((doc_id, False, error_msg, 0.0, {"doc_length": -1}))
        except Exception as read_e:
             error_msg = f"File read error for {src_path}: {read_e}"
             print(f"[Worker {pid}] ERROR: {error_msg}")
             worker_results.append((doc_id, False, error_msg, 0.0, {"doc_length": 0}))

    batch_end_time = time.time()
    mem_after_batch = process.memory_info().rss / (1024 * 1024)
    print(f"[Worker {pid}] Finished processing batch. Time taken: {batch_end_time - batch_start_time:.2f}s. Memory after batch: {mem_after_batch:.2f} MB")

    # Explicitly delete large objects before worker potentially exits
    del worker_nlp
    del text # Delete last text object
    gc.collect()
    mem_after_gc = process.memory_info().rss / (1024 * 1024)
    print(f"[Worker {pid}] Memory after GC: {mem_after_gc:.2f} MB")
    return worker_results

# --- CLI Parser ---
# (create_parser unchanged)
def create_parser() -> argparse.ArgumentParser:
    """Creates parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="Corpus processor",
        description="Processes all documents in a corpus on CPU using batch processing.",
    )
    parser.add_argument("--model", type=str, default="grc_proiel_trf", help="Name of the spaCy model to use.")
    parser.add_argument("--dest", type=str, default="dat/greek/processed_data/", help="Destination directory for processed .spacy files.")
    parser.add_argument("--src_index", type=str, default="dat/greek/cleaned_parsed_data/index.csv", help="Path to the CSV index file containing source document paths.")
    parser.add_argument("--wandb_user", type=str, default="sozialismus-au", help="Weights & Biases username or entity.")
    parser.add_argument("--wandb_project", type=str, default="model-tracking", help="Weights & Biases project name.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional name for the W&B run.")
    parser.add_argument("--sample_interval", type=int, default=50, help="Log detailed system/server metrics every N documents.") # Increased default
    parser.add_argument("--checkpoint_interval", type=int, default=200, help="Save checkpoint data every N documents.") # Increased default
    parser.add_argument("--batch_items", type=int, default=64, help="Number of documents to group into a single batch for worker processing.")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--force_sync", action="store_true", help="Force disk sync after each file save (SLOW, use for critical data only).")

    return parser

# --- Main Execution ---

def main():
    parser = create_parser()
    args = parser.parse_args()
    print(
        "--------------------------\n"
        "--- BATCH PROCESS CORPUS ---\n"
        "--------------------------\n"
    )

    # Creating destination directory
    # (Unchanged)
    dest_path_obj = Path(args.dest)
    print(f"Creating destination directory ({args.dest})")
    dest_path_obj.mkdir(exist_ok=True, parents=True)

    # Generate a descriptive run name if none provided
    # (Unchanged)
    run_name = args.run_name or f"corpus-processing-{args.model}-{time.strftime('%Y%m%d-%H%M%S')}"

    # Initialize wandb if not disabled
    # (Unchanged, logs SYSTEM_CONFIG which now includes N_PROCESS_PIPE etc.)
    if not args.disable_wandb:
        print(f"Initializing wandb with run name: {run_name}")
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_user,
                name=run_name,
                tags=["corpus-processing", args.model, "batch-mode"]
            )
            wandb.config.update(vars(args))
            wandb.config.update({
                "system_config": SYSTEM_CONFIG,
                "max_length": MAX_LENGTH,
                "n_process_pipe": N_PROCESS_PIPE, # Log the pipe process count
                "batch_size_pipe": BATCH_SIZE_PIPE, # Log the pipe batch size
                "device": "cpu",
            })
        except Exception as wandb_e:
             print(f"FATAL: Failed to initialize WandB: {wandb_e}")
             args.disable_wandb = True
             print("WandB logging has been disabled.")
    else:
        print("Weights & Biases logging disabled")

    # --- NUMA Setup ---
    # (Unchanged)
    numa_enabled_runtime = False
    if SYSTEM_CONFIG.get("NUMA_NODES", 1) > 1 and numa: # Check numa is available
        numa_enabled_runtime = set_numa_awareness()
        print(f"NUMA awareness runtime status: {'Enabled' if numa_enabled_runtime else 'Not available/active'}")
        if not args.disable_wandb:
             wandb.config.update({"numa_enabled_runtime": numa_enabled_runtime})

    # --- Model Loading (Test Load Only) ---
    # (Unchanged)
    print(f"Testing load of NLP model: {args.model}")
    start_time_model_load = time.time()
    try:
        temp_nlp = spacy.load(args.model)
        model_load_time = time.time() - start_time_model_load
        print(f"Model test load successful in {model_load_time:.2f} seconds")
        if not args.disable_wandb:
            wandb.log({"model_load_time_seconds": model_load_time})
            pipeline_components = [{"name": name, "type": str(type(component))}
                                   for name, component in temp_nlp.pipeline]
            wandb.log({"model_pipeline": wandb.Table(
                data=[[comp["name"], comp["type"]] for comp in pipeline_components],
                columns=["Component", "Type"]
            )})
        del temp_nlp
        gc.collect()
    except Exception as e:
        error_message = f"Failed to test load model: {str(e)}. Workers might fail."
        print(f"ERROR: {error_message}")
        if not args.disable_wandb:
            wandb.log({"critical_errors": wandb.Table(
                data=[["Model Test Loading", error_message]], columns=["Stage", "Error"] )})

    # --- Index Loading ---
    # (Unchanged)
    print(f"Loading index of source files from {args.src_index}")
    try:
        parsed_index = pd.read_csv(args.src_index)
        if 'document_id' not in parsed_index.columns or 'dest_path' not in parsed_index.columns:
             raise ValueError("Index CSV must contain 'document_id' and 'dest_path' columns.")
        parsed_index['document_id'] = parsed_index['document_id'].astype(str)
        n_total_in_index = len(parsed_index.index)
        print(f"Loaded index with {n_total_in_index} documents")
        if not args.disable_wandb: wandb.log({"total_documents_in_index": n_total_in_index})
    except Exception as e:
        error_message = f"Failed to load or validate index '{args.src_index}': {str(e)}"
        print(f"CRITICAL ERROR: {error_message}")
        if not args.disable_wandb: wandb.log({"critical_errors": wandb.Table( data=[["Index Loading", error_message]], columns=["Stage", "Error"] )})
        return

    # --- Filter Processed Files ---
    # (Unchanged)
    print(f"Checking for already processed files in {args.dest}...")
    done_ids = get_done_ids(args.dest)
    n_done = len(done_ids)
    if n_done > 0:
        done_filter = parsed_index.document_id.isin(done_ids)
        n_already_done_in_index = done_filter.sum()
        print(f"Found {n_done} existing .spacy files. Ignoring {n_already_done_in_index} matching entries in index.")
        parsed_index = parsed_index[~done_filter]
    else:
        print("No previously processed files found matching the index.")
    n_left = len(parsed_index.index)
    if n_left == 0:
        print("No documents left to process.")
        if not args.disable_wandb: wandb.finish()
        return
    print(f"Documents to process: {n_left}")
    if not args.disable_wandb:
        wandb.log({ "documents_to_process": n_left, "documents_already_processed": n_done,
                    "initial_progress": Plotly(progress_piechart(n_done, n_left + n_done)) })

    # --- Prepare Batches ---
    # (Unchanged)
    print(f"Preparing data batches (batch size: {args.batch_items})...")
    doc_data = list(zip(
        parsed_index.dest_path,
        parsed_index.document_id,
        parsed_index.document_id.map(lambda doc_id: str(dest_path_obj / f"{doc_id}.spacy"))
    ))
    num_batches = (n_left + args.batch_items - 1) // args.batch_items
    data_batches = []
    for i in range(0, n_left, args.batch_items):
        end = min(i + args.batch_items, n_left)
        data_batches.append(doc_data[i:end])
    print(f"Created {len(data_batches)} batches.")

    # --- Initialize Tracking ---
    # (Unchanged)
    start_time_total = time.time()
    processed_chars_total = 0
    successful_docs = 0
    failed_docs = 0
    total_processed_in_loop = 0
    checkpoint_dir = dest_path_obj / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_df = pd.DataFrame(columns=["document_id", "status", "processing_time", "timestamp"])

    # --- Main Processing Loop (Batch Mode) ---
    # (Logic for handling results and logging largely unchanged)
    print(f"Starting processing using {SYSTEM_CONFIG['NUMA_NODES']} worker processes...")
    batch_pbar = tqdm(total=len(data_batches), desc="Processing Batches", unit="batch")
    doc_pbar = tqdm(total=n_left, desc="Processed Documents", unit="doc")

    with ProcessPoolExecutor(max_workers=SYSTEM_CONFIG["NUMA_NODES"]) as executor:
        futures = {
             executor.submit(process_batch, batch, args.model, SYSTEM_CONFIG, args.disable_wandb): batch_idx
             for batch_idx, batch in enumerate(data_batches)
        }

        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                batch_results = future.result()
            except Exception as e:
                # Handle errors at the batch level (e.g., worker crash like before)
                print(f"CRITICAL ERROR: Processing batch {batch_idx} failed entirely: {e}")
                failed_count_in_batch = len(data_batches[batch_idx])
                failed_docs += failed_count_in_batch
                total_processed_in_loop += failed_count_in_batch
                doc_pbar.update(failed_count_in_batch)
                if not args.disable_wandb:
                   try: wandb.log({"batch_processing_error": str(e)})
                   except Exception as log_e: print(f"Error logging batch error: {log_e}")
                continue

            # --- Process results from the completed batch ---
            # (Result processing logic remains the same...)
            batch_start_index = batch_idx * args.batch_items
            for i, (doc_id, success, error_message, processing_time, metrics) in enumerate(batch_results):
                doc_index_in_run = total_processed_in_loop
                current_doc_global_index = n_done + doc_index_in_run
                total_processed_in_loop += 1
                doc_length = metrics.get("doc_length", 0)
                if doc_length > 0: processed_chars_total += doc_length

                if success:
                    successful_docs += 1
                    status = "success"
                else:
                    failed_docs += 1
                    status = "failed"
                    if not args.disable_wandb:
                        try:
                            wandb.log({"failed_documents": wandb.Table(
                                data=[[doc_id, error_message]], columns=["Document ID", "Error"] )})
                        except Exception as log_e: print(f"Error logging failed doc table: {log_e}")

                new_row = pd.DataFrame([{ "document_id": doc_id, "status": status,
                    "processing_time": processing_time, "timestamp": time.time() }])
                checkpoint_df = pd.concat([checkpoint_df, new_row], ignore_index=True)

                # --- Logging ---
                if not args.disable_wandb:
                    elapsed_time = time.time() - start_time_total
                    docs_per_second = total_processed_in_loop / elapsed_time if elapsed_time > 0 else 0
                    chars_per_second = processed_chars_total / elapsed_time if elapsed_time > 0 else 0
                    est_rem_time = (elapsed_time / total_processed_in_loop) * (n_left - total_processed_in_loop) if total_processed_in_loop > 0 else 0
                    log_data = {
                        "n_processed_total": current_doc_global_index, "n_processed_this_run": total_processed_in_loop,
                        "progress_percent": (current_doc_global_index / (n_left + n_done)) * 100 if (n_left + n_done) > 0 else 0,
                        "doc_length": doc_length, "processing_time_seconds": processing_time,
                        "processing_speed_chars_per_sec": doc_length / processing_time if processing_time > 0 else 0,
                        "cumulative_successful_docs": successful_docs, "cumulative_failed_docs": failed_docs,
                        "elapsed_time_minutes": elapsed_time / 60, "avg_docs_per_second": docs_per_second,
                        "avg_chars_per_second": chars_per_second, "estimated_remaining_time_minutes": est_rem_time / 60,
                        **metrics
                    }
                    if total_processed_in_loop % 100 == 0 or total_processed_in_loop == n_left:
                        log_data["progress_pie"] = Plotly(progress_piechart(current_doc_global_index, n_left + n_done))
                    try:
                        wandb.log(log_data)
                    except Exception as log_e: print(f"Error logging W&B data: {log_e}")

                # --- Periodic Tasks ---
                if total_processed_in_loop % args.checkpoint_interval == 0 and total_processed_in_loop > 0:
                    checkpoint_path = checkpoint_dir / f"checkpoint_{current_doc_global_index}.csv"
                    try:
                        checkpoint_df.to_csv(checkpoint_path, index=False)
                        print(f"\nSaved checkpoint ({len(checkpoint_df)} rows) at {checkpoint_path}")
                        if not args.disable_wandb:
                            checkpoint_artifact = wandb.Artifact(f"checkpoint_{current_doc_global_index}", type="checkpoint")
                            checkpoint_artifact.add_file(str(checkpoint_path))
                            wandb.log_artifact(checkpoint_artifact)
                    except Exception as cp_e: print(f"\nError saving checkpoint: {cp_e}")

                if total_processed_in_loop % args.sample_interval == 0 and total_processed_in_loop > 0:
                    print(f"\nLogging system/server metrics at doc {total_processed_in_loop}...")
                    log_system_metrics(args.disable_wandb)
                    log_server_metrics(args.disable_wandb)

                if total_processed_in_loop % 100 == 0 and total_processed_in_loop > 0: # Memory management frequency
                    print("\nRunning periodic memory management...")
                    manage_memory()

                doc_pbar.update(1) # Update doc progress bar here

            # Update batch progress bar after processing all results from a batch
            batch_pbar.update(1)

    batch_pbar.close()
    doc_pbar.close()
    print("\nProcessing loop finished.")


    # --- Final Index and Checkpoint ---
    # (Unchanged)
    print("Creating final index for processed documents...")
    final_run_results = checkpoint_df[['document_id', 'status']].copy()
    processed_subset_index = parsed_index[parsed_index['document_id'].isin(final_run_results['document_id'])].copy()
    final_index = pd.merge(processed_subset_index[['document_id', 'dest_path']], final_run_results, on='document_id', how='left')
    final_index.rename(columns={'dest_path': 'src_path', 'status': 'processing_status'}, inplace=True)
    final_index['dest_path'] = final_index['document_id'].map(lambda doc_id: str(dest_path_obj / f"{doc_id}.spacy"))
    index_path = dest_path_obj / "index_processed_run.csv"
    try:
        final_index.to_csv(index_path, index=False)
        print(f"Saved index for this run ({len(final_index)} entries) to {index_path}")
        if not args.disable_wandb:
            index_artifact = wandb.Artifact(f"processed_corpus_index_{run_name}", type="dataset_index")
            index_artifact.add_file(str(index_path))
            wandb.log_artifact(index_artifact)
    except Exception as idx_e: print(f"Error saving final index: {idx_e}")

    final_checkpoint_path = checkpoint_dir / "final_checkpoint_run.csv"
    try:
        checkpoint_df.to_csv(final_checkpoint_path, index=False)
        print(f"Saved final run checkpoint ({len(checkpoint_df)} rows) to {final_checkpoint_path}")
        if not args.disable_wandb:
            final_checkpoint_artifact = wandb.Artifact(f"final_checkpoint_{run_name}", type="checkpoint")
            final_checkpoint_artifact.add_file(str(final_checkpoint_path))
            wandb.log_artifact(final_checkpoint_artifact)
    except Exception as fcp_e: print(f"Error saving final checkpoint: {fcp_e}")


    # --- Summary ---
    processing_time_total = time.time() - start_time_total
    actual_processed_count = successful_docs + failed_docs

    # --- W&B Logging FIX ---
    success_rate_float = (successful_docs / actual_processed_count * 100) if actual_processed_count > 0 else 0.0
    success_rate_str = f"{success_rate_float:.2f}%" # Keep string format for printout

    summary_data = [
        ["Total Documents in Index", n_total_in_index],
        ["Documents Already Processed (on disk)", n_done],
        ["Documents Attempted in This Run", actual_processed_count],
        ["Successfully Processed (this run)", successful_docs],
        ["Failed (this run)", failed_docs],
        ["Success Rate (this run)", success_rate_float], # Use float for W&B Table
        ["Total Processing Time (min)", processing_time_total / 60],
        ["Avg Time per Document (sec, this run)", (processing_time_total / actual_processed_count) if actual_processed_count > 0 else 0.0],
        ["Avg Processing Speed (chars/sec, this run)", (processed_chars_total / processing_time_total) if processing_time_total > 0 else 0.0],
        ["Total Characters Processed (this run)", processed_chars_total],
        ["Batch Size (Items per Worker Job)", args.batch_items],
        ["Worker Processes (NUMA Nodes)", SYSTEM_CONFIG['NUMA_NODES']],
        ["spaCy Processes per Worker (nlp.pipe)", N_PROCESS_PIPE], # Use correct name
        ["spaCy Batch Size per Worker (nlp.pipe)", BATCH_SIZE_PIPE], # Use correct name
        ["Max Document Length (chars)", MAX_LENGTH],
    ]

    # Print summary using the string format for success rate
    print("\n===== Processing Summary =====")
    for name, value in summary_data:
        if name == "Success Rate (this run)":
             print(f"{name}: {success_rate_str}") # Print formatted string
        elif isinstance(value, float):
             print(f"{name}: {value:.2f}") # Format floats for printing
        else:
             print(f"{name}: {value}")


    # Log summary to W&B using the float for success rate
    if not args.disable_wandb:
        try:
            # Ensure data types match W&B expectations (Numbers for numeric fields)
            summary_table_data = [[item[0], item[1]] for item in summary_data]
            wandb.log({"final_summary": wandb.Table(
                data=summary_table_data,
                columns=["Metric", "Value"]
            )})
            wandb.finish()
            print("\nWandB run finished.")
        except Exception as log_e:
            print(f"Error logging final summary / finishing WandB: {log_e}") # Original error message was helpful

if __name__ == "__main__":
    main()
