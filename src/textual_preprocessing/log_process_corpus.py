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
except ImportError:
    numa = None

# --- System Resource Configuration ---

def configure_system_resources():
    """Configure system resources optimized for high-core-count Xeon servers"""
    cpu_count = os.cpu_count()
    available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)

    # For high-core-count servers, use 50% of cores for better throughput
    # while leaving resources for the OS and other processes (used by nlp.pipe within workers)
    n_process = max(1, int(cpu_count * 0.50))

    # Larger batch sizes for high-core systems (used by nlp.pipe within workers)
    batch_size = max(16, min(int(available_memory_gb * 3), 128))

    # Scale max length more aggressively with available memory
    # Increased cap from 10k to 20k as per suggestion
    max_length = min(2 * 10**4, int(available_memory_gb * 8 * 10**3))
    # Ensure max_length is reasonable, fallback to spaCy default if calculated value is too low or high
    max_length = max(10000, min(max_length, 2 * 10**6)) # Keep within a sane range (10k to 2M)

    # Add NUMA awareness (assuming dual-socket Xeon Gold 6130 as per prompt)
    # This should ideally be detected, but hardcoding for the specific case
    numa_nodes = 2 # Defaulting to 2 based on prompt info

    # More threads for PyTorch tensor operations in workers
    torch_threads = max(4, cpu_count // 8)

    print(f"System Config: CPU={cpu_count}, MemGB={available_memory_gb:.2f}, CoresPerWorker={n_process}, PipeBatch={batch_size}, MaxLen={max_length}, TorchThreads={torch_threads}, NUMANodes={numa_nodes}")

    return {
        "MAX_LENGTH": max_length,
        "N_PROCESS": n_process, # For nlp.pipe inside worker
        "BATCH_SIZE": batch_size, # For nlp.pipe inside worker
        "CPU_COUNT": cpu_count,
        "AVAILABLE_MEMORY_GB": available_memory_gb,
        "TORCH_THREADS": torch_threads, # For workers
        "NUMA_NODES": numa_nodes # For ProcessPoolExecutor
    }

# Get system-specific parameters
SYSTEM_CONFIG = configure_system_resources()
MAX_LENGTH = SYSTEM_CONFIG["MAX_LENGTH"]
N_PROCESS = SYSTEM_CONFIG["N_PROCESS"] # Used inside process_document's nlp.pipe call
BATCH_SIZE = SYSTEM_CONFIG["BATCH_SIZE"] # Used inside process_document's nlp.pipe call


# --- NUMA Awareness Setup ---

def set_numa_awareness():
    """Set NUMA awareness if available"""
    numa_available = False
    try:
        if numexpr:
            # Set numexpr to use multiple threads - good for Xeon processors
            numexpr.set_num_threads(SYSTEM_CONFIG["TORCH_THREADS"])
            print(f"NumExpr using {SYSTEM_CONFIG['TORCH_THREADS']} threads.")

        # If numa library available, use it
        if numa:
            if numa.available():
                # Let OS handle NUMA scheduling, seems generally preferred
                # Or try numa.set_interleave_mask(numa.all_nodes_mask) for interleaving
                # numa.set_preferred(None) # May not be needed if relying on OS scheduler + ProcessPoolExecutor affinity
                print("NUMA library available.")
                numa_available = True
            else:
                print("NUMA library loaded but not available on this system.")
        else:
             print("NUMA library not installed.")
        return numa_available
    except Exception as e:
        print(f"Error setting NUMA awareness: {e}")
        return False

# --- Constants ---

TOKEN_ATTRS = [ # Consider reducing this list if not all attributes are needed
    "IS_ALPHA", "IS_ASCII", "IS_DIGIT", "IS_LOWER", "IS_PUNCT", "IS_SPACE",
    "IS_TITLE", "IS_UPPER", "LIKE_URL", "LIKE_NUM", "LIKE_EMAIL", "IS_STOP",
    "IS_QUOTE", "IS_BRACKET", "IS_LEFT_PUNCT", "IS_RIGHT_PUNCT", "IS_CURRENCY",
    "ID", "ORTH", "LOWER", "NORM", "SHAPE", "PREFIX", "SUFFIX", "LENGTH",
    "LEMMA", "POS", "TAG", "DEP", "ENT_IOB", "ENT_TYPE", "ENT_ID", "ENT_KB_ID",
    "HEAD", "SENT_START", "SPACY", "LANG", "MORPH", "IDX",
]

# --- Helper Functions ---

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
    if disable_wandb or not numa: return # Skip if wandb disabled or numa not available
    try:
        server_log = {}
        # Log NUMA-specific metrics
        if numa and numa.available():
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
            # Limit logged cores if there are too many? e.g. only log first 64?
            # core_usage = {f"core_{i}": percent for i, percent in enumerate(per_cpu_percent[:64])}
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
    try:
        collected = gc.collect()
        print(f"Garbage Collector: Freed {collected} objects.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA cache.")
    except Exception as e:
        print(f"Error during memory management: {e}")


def process_document(text: str, nlp: Language, dest: str, doc_id: str = None, disable_wandb: bool = False) -> Tuple[bool, str, float, Dict]:
    """Turns text into a spaCy document, optimized for high-core-count systems.
       Uses nlp.pipe internally for parallelism within the worker process.

    Returns:
        Tuple containing (success status, error message if any, processing time, metrics dict)
    """
    metrics = {
        "doc_length": len(text),
        "had_to_split": len(text) > MAX_LENGTH,
    }
    start_time = time.time()

    try:
        # TORCH_THREADS should be set per process (done in worker)
        # torch.set_num_threads(SYSTEM_CONFIG["TORCH_THREADS"]) # Set in worker instead

        # Track memory usage within this specific document processing call if needed
        # peak_memory_before = psutil.Process().memory_info().rss / (1024 * 1024)

        segment_size = MAX_LENGTH # Default segment size
        if len(text) > MAX_LENGTH:
            # Use smaller segments for more parallelism on high-core machines
            # Decision based on global config, execution happens here.
            segment_size = MAX_LENGTH // 2 if SYSTEM_CONFIG["CPU_COUNT"] > 32 else MAX_LENGTH
            texts = split_text_on_full_stop(text, segment_size)
            metrics["num_segments"] = len(texts)
            metrics["avg_segment_length"] = sum(len(t) for t in texts) / len(texts) if texts else 0
            metrics["segment_size_used"] = segment_size # Log the actual size used

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

        # Use nlp.pipe() to process texts using configured parallelism within this worker
        pipe_start_time = time.time()
        # N_PROCESS and BATCH_SIZE are from the global SYSTEM_CONFIG
        docs = list(nlp.pipe(texts, n_process=N_PROCESS, batch_size=BATCH_SIZE))
        pipe_time = time.time() - pipe_start_time
        metrics["pipe_processing_time"] = pipe_time

        # Track memory after processing
        # peak_memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
        # metrics["memory_usage_mb"] = peak_memory_after - peak_memory_before

        # Combine the processed segments back into a single document
        combine_start_time = time.time()
        if len(docs) > 1:
            doc = Doc.from_docs(docs)
        elif len(docs) == 1:
            doc = docs[0]
        else:
            # Handle case where splitting resulted in no processable segments
            # Create an empty doc
            vocab = nlp.vocab # Get vocab from the nlp object
            doc = Doc(vocab)
            print(f"Warning: No segments processed for doc_id {doc_id}, creating empty Doc.")
        combine_time = time.time() - combine_start_time
        metrics["doc_combine_time"] = combine_time

        # Collect document statistics
        metrics["token_count"] = len(doc)
        metrics["entity_count"] = len(doc.ents)
        try:
            # Sentence counting can fail on empty or unusual docs
            metrics["sentence_count"] = len(list(doc.sents)) if len(doc) > 0 else 0
        except Exception as sent_e:
            print(f"Warning: Could not count sentences for {doc_id}: {sent_e}")
            metrics["sentence_count"] = 0

        # Save document
        save_start_time = time.time()
        save_document(doc, dest=dest, disable_wandb=disable_wandb) # Pass wandb flag
        save_time = time.time() - save_start_time
        metrics["doc_save_time"] = save_time

        processing_time = time.time() - start_time
        metrics["total_processing_time"] = processing_time

        return True, "", processing_time, metrics
    except Exception as e:
        error_message = f"Error in process_document for {doc_id}: {str(e)}"
        print(error_message) # Print error for immediate visibility
        if not disable_wandb:
             try:
                 wandb.log({"processing_errors": wandb.Table(
                     data=[[doc_id, dest, str(e)]], # Use original error for logging
                     columns=["Document ID", "Document Path", "Error Message"]
                 )})
             except Exception as log_e: print(f"Error logging processing error: {log_e}")
        # Return metrics gathered so far, even on failure
        metrics["total_processing_time"] = time.time() - start_time
        return False, str(e), metrics.get("total_processing_time", 0.0), metrics


# --- Worker Function for ProcessPoolExecutor ---

def process_batch(batch_paths_ids_dests, nlp_model_name, system_config, disable_wandb):
    """Worker function to process a batch of documents."""
    worker_results = []
    worker_nlp = None
    pid = os.getpid()
    print(f"[Worker {pid}] Loading model '{nlp_model_name}'...")

    try:
        # Set PyTorch threads for this worker process
        torch.set_num_threads(system_config["TORCH_THREADS"])
        print(f"[Worker {pid}] Set torch threads to {system_config['TORCH_THREADS']}")

        # Load model in each worker process
        worker_nlp = spacy.load(nlp_model_name)
        worker_nlp.max_length = system_config["MAX_LENGTH"]
        print(f"[Worker {pid}] Model loaded. Max length: {worker_nlp.max_length}")

    except Exception as load_e:
        print(f"FATAL ERROR in worker {pid}: Could not load model {nlp_model_name}: {load_e}")
        # Return failure for all items in this batch
        error_msg = f"Worker {pid} failed to load model: {load_e}"
        for _, doc_id, _ in batch_paths_ids_dests:
             worker_results.append((doc_id, False, error_msg, 0.0, {"doc_length": 0}))
        return worker_results # Exit early if model load fails

    print(f"[Worker {pid}] Processing batch of {len(batch_paths_ids_dests)} documents...")
    for i, (src_path, doc_id, dest_path) in enumerate(batch_paths_ids_dests):
        text = None
        try:
            # Read the text file
            with open(src_path, 'r', encoding='utf-8') as f:
                text = f.read()
            if not text:
                 print(f"[Worker {pid}] Warning: Document {doc_id} ({src_path}) is empty.")
                 # Process empty text to create an empty .spacy file
                 # Fallthrough to process_document which handles empty text

            # Process the document using the main function
            success, error, proc_time, metrics = process_document(
                text if text is not None else "", # Ensure text is not None
                worker_nlp,
                dest_path,
                doc_id,
                disable_wandb=disable_wandb
            )
            worker_results.append((doc_id, success, error, proc_time, metrics))
            # Optional: print progress within worker
            # if (i + 1) % 10 == 0:
            #    print(f"[Worker {pid}] Processed {i+1}/{len(batch_paths_ids_dests)} in batch.")

        except FileNotFoundError:
             error_msg = f"File not found: {src_path}"
             print(f"[Worker {pid}] ERROR: {error_msg}")
             worker_results.append((doc_id, False, error_msg, 0.0, {"doc_length": -1})) # Indicate file not found
        except Exception as read_e:
             error_msg = f"File read error for {src_path}: {read_e}"
             print(f"[Worker {pid}] ERROR: {error_msg}")
             # Pass metrics dict even on read failure, length might be unknown
             worker_results.append((doc_id, False, error_msg, 0.0, {"doc_length": 0}))

    print(f"[Worker {pid}] Finished processing batch.")
    return worker_results


# --- CLI Parser ---

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
    dest_path_obj = Path(args.dest)
    print(f"Creating destination directory ({args.dest})")
    dest_path_obj.mkdir(exist_ok=True, parents=True)

    # Generate a descriptive run name if none provided
    run_name = args.run_name or f"corpus-processing-{args.model}-{time.strftime('%Y%m%d-%H%M%S')}"

    # Initialize wandb if not disabled
    if not args.disable_wandb:
        print(f"Initializing wandb with run name: {run_name}")
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_user,
                name=run_name,
                tags=["corpus-processing", args.model, "batch-mode"]
            )
            # Log system configuration and arguments
            wandb.config.update(vars(args)) # Log command line args
            wandb.config.update({
                "system_config": SYSTEM_CONFIG, # Log the determined config
                "max_length": MAX_LENGTH, # Redundant but explicit
                "n_process_worker": N_PROCESS,
                "batch_size_worker": BATCH_SIZE,
                "device": "cpu",
            })
        except Exception as wandb_e:
             print(f"FATAL: Failed to initialize WandB: {wandb_e}")
             args.disable_wandb = True # Force disable if init fails
             print("WandB logging has been disabled.")
    else:
        print("Weights & Biases logging disabled")

    # --- NUMA Setup ---
    numa_enabled_runtime = False
    if SYSTEM_CONFIG.get("NUMA_NODES", 1) > 1:
        numa_enabled_runtime = set_numa_awareness()
        print(f"NUMA awareness runtime status: {'Enabled' if numa_enabled_runtime else 'Not available/active'}")
        if not args.disable_wandb:
             wandb.config.update({"numa_enabled_runtime": numa_enabled_runtime})

    # --- Model Loading (Test Load Only) ---
    # Actual model loading happens in worker processes
    print(f"Testing load of NLP model: {args.model}")
    start_time_model_load = time.time()
    try:
        # Load briefly to check availability and log pipeline
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
        del temp_nlp # Release memory
        gc.collect()
    except Exception as e:
        error_message = f"Failed to test load model: {str(e)}. Workers might fail."
        print(f"ERROR: {error_message}")
        if not args.disable_wandb:
            wandb.log({"critical_errors": wandb.Table(
                data=[["Model Test Loading", error_message]],
                columns=["Stage", "Error"]
            )})
        # Decide whether to exit or continue (workers might still succeed if it was transient)
        # return # Exit if model load test fails

    # --- Index Loading ---
    print(f"Loading index of source files from {args.src_index}")
    try:
        parsed_index = pd.read_csv(args.src_index) # index_col=0 removed assuming standard CSV
        # Check required columns exist
        if 'document_id' not in parsed_index.columns or 'dest_path' not in parsed_index.columns:
             raise ValueError("Index CSV must contain 'document_id' and 'dest_path' columns.")
        parsed_index['document_id'] = parsed_index['document_id'].astype(str) # Ensure ID is string
        n_total_in_index = len(parsed_index.index)
        print(f"Loaded index with {n_total_in_index} documents")

        if not args.disable_wandb:
            wandb.log({"total_documents_in_index": n_total_in_index})
    except Exception as e:
        error_message = f"Failed to load or validate index '{args.src_index}': {str(e)}"
        print(f"CRITICAL ERROR: {error_message}")
        if not args.disable_wandb:
            wandb.log({"critical_errors": wandb.Table(
                data=[["Index Loading", error_message]],
                columns=["Stage", "Error"]
            )})
        return # Cannot proceed without index

    # --- Filter Processed Files ---
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
        wandb.log({
            "documents_to_process": n_left,
            "documents_already_processed": n_done, # Total found on disk
            "initial_progress": Plotly(progress_piechart(n_done, n_left + n_done)) # Pie based on index count + disk count
        })

    # --- Prepare Batches ---
    print(f"Preparing data batches (batch size: {args.batch_items})...")
    # Each item: (source_file_path, document_id, destination_spacy_path)
    doc_data = list(zip(
        parsed_index.dest_path, # Source paths from index column named 'dest_path'
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
    start_time_total = time.time()
    processed_chars_total = 0
    successful_docs = 0
    failed_docs = 0
    total_processed_in_loop = 0 # Counter for docs processed in this run

    # Create checkpoint directory
    checkpoint_dir = dest_path_obj / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Checkpoint dataframe for tracking processed documents in this run
    checkpoint_df = pd.DataFrame(columns=["document_id", "status", "processing_time", "timestamp"])

    # --- Main Processing Loop (Batch Mode) ---
    print(f"Starting processing using {SYSTEM_CONFIG['NUMA_NODES']} worker processes...")
    # Progress bar for batches
    batch_pbar = tqdm(total=len(data_batches), desc="Processing Batches", unit="batch")
    # Progress bar for documents
    doc_pbar = tqdm(total=n_left, desc="Processed Documents", unit="doc")

    with ProcessPoolExecutor(max_workers=SYSTEM_CONFIG["NUMA_NODES"]) as executor:
        # Submit all batches
        # Pass necessary config items explicitly to avoid reliance on potentially non-picklable globals
        futures = {
             executor.submit(process_batch, batch, args.model, SYSTEM_CONFIG, args.disable_wandb): batch_idx
             for batch_idx, batch in enumerate(data_batches)
        }

        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                batch_results = future.result() # List of tuples: (doc_id, success, error, time, metrics)
            except Exception as e:
                # Handle errors at the batch level (e.g., worker crash)
                print(f"CRITICAL ERROR: Processing batch {batch_idx} failed entirely: {e}")
                # Estimate failed docs and update progress - difficult, depends how many were processed before crash
                # For simplicity, log the batch error and update doc progress bar minimally
                failed_count_in_batch = len(data_batches[batch_idx])
                failed_docs += failed_count_in_batch
                total_processed_in_loop += failed_count_in_batch # Count them as 'processed' (attempted)
                doc_pbar.update(failed_count_in_batch)
                if not args.disable_wandb:
                   try: wandb.log({"batch_processing_error": str(e)})
                   except Exception as log_e: print(f"Error logging batch error: {log_e}")
                continue # Skip processing results for this failed batch

            # --- Process results from the completed batch ---
            batch_start_index = batch_idx * args.batch_items # Estimate start index for logging steps

            for i, (doc_id, success, error_message, processing_time, metrics) in enumerate(batch_results):
                doc_index_in_run = total_processed_in_loop # Index relative to start of this run (0 to n_left-1)
                # For logging steps, use global index (includes previously done)
                current_doc_global_index = n_done + doc_index_in_run

                # Update overall counters
                total_processed_in_loop += 1
                doc_length = metrics.get("doc_length", 0)
                if doc_length > 0: # Only count chars if file was read
                    processed_chars_total += doc_length

                # Update success/failure counts and status
                if success:
                    successful_docs += 1
                    status = "success"
                else:
                    failed_docs += 1
                    status = "failed"
                    if not args.disable_wandb:
                        # Log specific failure
                        try:
                            wandb.log({"failed_documents": wandb.Table(
                                data=[[doc_id, error_message]],
                                columns=["Document ID", "Error"]
                            )})
                        except Exception as log_e: print(f"Error logging failed doc table: {log_e}")

                # Add to checkpoint dataframe (use concat for efficiency)
                new_row = pd.DataFrame([{
                    "document_id": doc_id, "status": status,
                    "processing_time": processing_time, "timestamp": time.time()
                }])
                checkpoint_df = pd.concat([checkpoint_df, new_row], ignore_index=True)

                # --- Logging ---
                if not args.disable_wandb:
                    elapsed_time = time.time() - start_time_total
                    docs_per_second = total_processed_in_loop / elapsed_time if elapsed_time > 0 else 0
                    chars_per_second = processed_chars_total / elapsed_time if elapsed_time > 0 else 0
                    estimated_remaining_time = (elapsed_time / total_processed_in_loop) * (n_left - total_processed_in_loop) if total_processed_in_loop > 0 else 0

                    log_data = {
                        "n_processed_total": current_doc_global_index, # Global count
                        "n_processed_this_run": total_processed_in_loop, # Count for this run
                        "progress_percent": (current_doc_global_index / (n_left + n_done)) * 100 if (n_left + n_done) > 0 else 0,
                        # "progress_pie": Plotly(progress_piechart(current_doc_global_index, n_left + n_done)), # Plotly can be slow, log less often?
                        "doc_length": doc_length,
                        "processing_time_seconds": processing_time,
                        "processing_speed_chars_per_sec": doc_length / processing_time if processing_time > 0 else 0,
                        "cumulative_successful_docs": successful_docs,
                        "cumulative_failed_docs": failed_docs,
                        "elapsed_time_minutes": elapsed_time / 60,
                        "avg_docs_per_second": docs_per_second,
                        "avg_chars_per_second": chars_per_second,
                        "estimated_remaining_time_minutes": estimated_remaining_time / 60,
                        **metrics # Include metrics from process_document
                    }
                    # Only log pie chart occasionally
                    if total_processed_in_loop % 100 == 0 or total_processed_in_loop == n_left:
                        log_data["progress_pie"] = Plotly(progress_piechart(current_doc_global_index, n_left + n_done))

                    try:
                        wandb.log(log_data) # Use default step increment or specify step=current_doc_global_index
                    except Exception as log_e: print(f"Error logging W&B data: {log_e}")

                # --- Periodic Tasks (Checkpointing, System Metrics, Memory Management) ---
                if total_processed_in_loop % args.checkpoint_interval == 0 and total_processed_in_loop > 0:
                    checkpoint_path = checkpoint_dir / f"checkpoint_{current_doc_global_index}.csv"
                    try:
                        checkpoint_df.to_csv(checkpoint_path, index=False)
                        print(f"\nSaved checkpoint ({len(checkpoint_df)} rows) at {checkpoint_path}")
                        if not args.disable_wandb:
                            # Log artifact
                            checkpoint_artifact = wandb.Artifact(f"checkpoint_{current_doc_global_index}", type="checkpoint")
                            checkpoint_artifact.add_file(str(checkpoint_path))
                            wandb.log_artifact(checkpoint_artifact)
                    except Exception as cp_e:
                         print(f"\nError saving checkpoint: {cp_e}")

                if total_processed_in_loop % args.sample_interval == 0 and total_processed_in_loop > 0:
                    # Log system and server metrics
                    print(f"\nLogging system/server metrics at doc {total_processed_in_loop}...")
                    log_system_metrics(args.disable_wandb)
                    log_server_metrics(args.disable_wandb) # Logs NUMA/core stats if available

                    # NLP stats logging requires re-processing or passing Doc object.
                    # Commented out for batch mode efficiency.
                    # if success and text is not None: # Need text
                    #     try:
                    #         # Re-process a sample for stats (might be slow)
                    #         print(f"Logging NLP stats for sample of doc {doc_id}...")
                    #         temp_nlp_main = spacy.load(args.model) # Load model in main process just for this
                    #         sample_text = text[:min(len(text), MAX_LENGTH)]
                    #         sample_doc = temp_nlp_main(sample_text)
                    #         log_nlp_statistics(sample_doc, current_doc_global_index, doc_id, args.disable_wandb)
                    #         del temp_nlp_main # cleanup
                    #         gc.collect()
                    #     except Exception as e:
                    #          print(f"\nError logging NLP stats: {e}")
                    #          if not args.disable_wandb: wandb.log({"statistics_errors": str(e)})

                # Memory Management Call (Suggestion 5) - run fairly often
                if total_processed_in_loop % 100 == 0 and total_processed_in_loop > 0:
                    print("\nRunning periodic memory management...")
                    manage_memory()

                # Update document progress bar
                doc_pbar.update(1)

            # Update batch progress bar after processing all results from a batch
            batch_pbar.update(1)

    batch_pbar.close()
    doc_pbar.close()
    print("\nProcessing loop finished.")

    # --- Final Index and Checkpoint ---
    print("Creating final index for processed documents...")
    # Use the checkpoint_df which contains results from this run
    final_run_results = checkpoint_df[['document_id', 'status']].copy()
    # Merge with the original subset of the index that was processed
    processed_subset_index = parsed_index[parsed_index['document_id'].isin(final_run_results['document_id'])].copy()
    # Add source path back if needed, assuming 'dest_path' in parsed_index was the source
    final_index = pd.merge(processed_subset_index[['document_id', 'dest_path']], final_run_results, on='document_id', how='left')
    final_index.rename(columns={'dest_path': 'src_path', 'status': 'processing_status'}, inplace=True)
    # Add the destination path column
    final_index['dest_path'] = final_index['document_id'].map(lambda doc_id: str(dest_path_obj / f"{doc_id}.spacy"))

    index_path = dest_path_obj / "index_processed_run.csv" # Save index for this run specifically
    try:
        final_index.to_csv(index_path, index=False)
        print(f"Saved index for this run ({len(final_index)} entries) to {index_path}")
        if not args.disable_wandb:
            index_artifact = wandb.Artifact(f"processed_corpus_index_{run_name}", type="dataset_index")
            index_artifact.add_file(str(index_path))
            wandb.log_artifact(index_artifact)
    except Exception as idx_e:
        print(f"Error saving final index: {idx_e}")


    # Save final checkpoint (contains all results from this run)
    final_checkpoint_path = checkpoint_dir / "final_checkpoint_run.csv"
    try:
        checkpoint_df.to_csv(final_checkpoint_path, index=False)
        print(f"Saved final run checkpoint ({len(checkpoint_df)} rows) to {final_checkpoint_path}")
        if not args.disable_wandb:
            final_checkpoint_artifact = wandb.Artifact(f"final_checkpoint_{run_name}", type="checkpoint")
            final_checkpoint_artifact.add_file(str(final_checkpoint_path))
            wandb.log_artifact(final_checkpoint_artifact)
    except Exception as fcp_e:
        print(f"Error saving final checkpoint: {fcp_e}")

    # --- Summary ---
    processing_time_total = time.time() - start_time_total
    actual_processed_count = successful_docs + failed_docs # Should equal total_processed_in_loop

    summary_data = [
        ["Total Documents in Index", n_total_in_index],
        ["Documents Already Processed (on disk)", n_done],
        ["Documents Attempted in This Run", actual_processed_count],
        ["Successfully Processed (this run)", successful_docs],
        ["Failed (this run)", failed_docs],
        ["Success Rate (this run)", f"{(successful_docs / actual_processed_count * 100):.2f}%" if actual_processed_count > 0 else "N/A"],
        ["Total Processing Time (min)", f"{processing_time_total / 60:.2f}"],
        ["Avg Time per Document (sec, this run)", f"{processing_time_total / actual_processed_count:.2f}" if actual_processed_count > 0 else "N/A"],
        ["Avg Processing Speed (chars/sec, this run)", f"{processed_chars_total / processing_time_total:.2f}" if processing_time_total > 0 else "N/A"],
        ["Total Characters Processed (this run)", processed_chars_total],
        ["Batch Size (Items per Worker Job)", args.batch_items],
        ["Worker Processes (NUMA Nodes)", SYSTEM_CONFIG['NUMA_NODES']],
        ["spaCy Processes per Worker (nlp.pipe)", N_PROCESS],
        ["spaCy Batch Size per Worker (nlp.pipe)", BATCH_SIZE],
        ["Max Document Length (chars)", MAX_LENGTH],
    ]

    # Print summary
    print("\n===== Processing Summary =====")
    for name, value in summary_data:
        print(f"{name}: {value}")

    # Log summary to W&B
    if not args.disable_wandb:
        try:
            wandb.log({"final_summary": wandb.Table(
                data=summary_data,
                columns=["Metric", "Value"]
            )})
            wandb.finish()
            print("\nWandB run finished.")
        except Exception as log_e:
            print(f"Error logging final summary / finishing WandB: {log_e}")

if __name__ == "__main__":
    main()
