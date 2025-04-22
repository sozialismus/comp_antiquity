# -*- coding: utf-8 -*-
"""Script responsible for cleaning/processing the corpus.
   (Reverted to single model load, using nlp.pipe for internal parallelism)
"""
import argparse
import gc
import glob
# import multiprocessing # Not explicitly needed now
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict
# Removed ProcessPoolExecutor imports

import pandas as pd
import plotly.graph_objects as go
import psutil
import spacy
import torch
import wandb
from spacy.language import Language
from spacy.tokens import Doc, DocBin
from tqdm import tqdm
# Assuming utils.streams.stream_files reads files lazily
from utils.streams import stream_files # Re-enabled stream_files
from wandb.data_types import Plotly

# Try importing NUMA libraries (same logic as before)
try: import numexpr
except ImportError: numexpr = None
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

# --- System Resource Configuration ---

def configure_system_resources():
    """Configure system resources for single model load, using nlp.pipe parallelism."""
    cpu_count = os.cpu_count()
    available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)

    # N_PROCESS: For nlp.pipe. Use a significant portion of cores now,
    # as the model is loaded only once. Start with 50% as per Xeon suggestion.
    # Reduce this if OOM occurs *during* nlp.pipe processing.
    n_process = max(1, int(cpu_count * 0.50)) # e.g., 32 on 64-core

    # BATCH_SIZE: For nlp.pipe. Keep memory-scaled logic.
    batch_size = max(16, min(int(available_memory_gb * 3), 128))

    # MAX_LENGTH: Keep improved logic
    max_length = min(2 * 10**4, int(available_memory_gb * 8 * 10**3))
    max_length = max(10000, min(max_length, 2 * 10**6)) # Sane range

    # NUMA_NODES: Info only, not directly used for workers now
    numa_nodes = 1 # Default if detection fails
    if numa:
        try:
            num_nodes_detected = numa.num_configured_nodes()
            if num_nodes_detected > 0: numa_nodes = num_nodes_detected
            print(f"Detected {numa_nodes} NUMA nodes.")
        except Exception as e:
            print(f"Could not detect NUMA nodes via library, using default {numa_nodes}. Error: {e}")

    # TORCH_THREADS: For the main process (and potentially inherited by nlp.pipe workers)
    # Can use more threads now. Let's try the original suggestion.
    torch_threads = max(4, cpu_count // 8) # e.g., 8 threads

    print(f"System Config: CPU={cpu_count}, MemGB={available_memory_gb:.2f}, PipeProcesses={n_process}, PipeBatch={batch_size}, MaxLen={max_length}, TorchThreads={torch_threads}, NUMANodes={numa_nodes}")

    return {
        "MAX_LENGTH": max_length,
        "N_PROCESS": n_process,         # Back to N_PROCESS for nlp.pipe
        "BATCH_SIZE": batch_size,       # Back to BATCH_SIZE for nlp.pipe
        "CPU_COUNT": cpu_count,
        "AVAILABLE_MEMORY_GB": available_memory_gb,
        "TORCH_THREADS": torch_threads,
        "NUMA_NODES": numa_nodes # Informational
    }

# Get system-specific parameters
SYSTEM_CONFIG = configure_system_resources()
MAX_LENGTH = SYSTEM_CONFIG["MAX_LENGTH"]
N_PROCESS = SYSTEM_CONFIG["N_PROCESS"]   # For nlp.pipe
BATCH_SIZE = SYSTEM_CONFIG["BATCH_SIZE"] # For nlp.pipe

# --- NUMA Awareness Setup ---
# (set_numa_awareness function unchanged)
def set_numa_awareness():
    """Set NUMA awareness if available"""
    numa_available = False
    try:
        if numexpr:
            numexpr.set_num_threads(SYSTEM_CONFIG["TORCH_THREADS"])
            print(f"NumExpr using {SYSTEM_CONFIG['TORCH_THREADS']} threads.")
        if numa:
            print("NUMA library available and supported.")
            numa_available = True
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
#  log_server_metrics, manage_memory functions remain unchanged)
# ... (paste unchanged helper functions here) ...
def get_done_ids(path: str) -> List[str]:
    """Finds documents that have already been cleaned using pathlib"""
    dest_path = Path(path)
    ids = []
    if dest_path.is_dir():
        ids = [p.stem for p in dest_path.glob("*.spacy")]
    return ids

def progress_piechart(n_processed: int, n_total: int) -> go.Figure:
    """Draws piechart of progress"""
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

def force_sync_directory(directory_path):
    """Force system to sync directory to disk (use RARELY)."""
    print("Warning: force_sync_directory called. This can severely impact performance.")
    try:
        directory = Path(directory_path)
        if not directory.is_dir(): return
        if hasattr(os, 'sync'): os.sync()
        elif os.name == 'nt':
            import ctypes
            ctypes.windll.kernel32.FlushFileBuffers(ctypes.c_void_p(-1))
        if hasattr(os, 'fsync'):
            try:
                if directory.exists():
                    fd = os.open(str(directory), os.O_RDONLY)
                    os.fsync(fd); os.close(fd)
            except OSError as e: print(f"Warning: Could not fsync directory {directory}: {e}")
            except Exception as e: print(f"Warning: General error fsyncing directory {directory}: {e}")
    except Exception as e: print(f"Warning: Could not force sync to disk: {e}")

def save_document(doc: Doc, dest: str, disable_wandb: bool, force_sync: bool = False) -> None:
    """Serializes and saves spaCy Document."""
    start_time = time.time(); success = True; error_msg = ""
    dest_path = Path(dest)
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        doc_bin = DocBin(attrs=TOKEN_ATTRS, docs=[doc])
        temp_dest = dest_path.with_suffix(dest_path.suffix + ".tmp")
        doc_bin.to_disk(str(temp_dest))
        temp_dest.rename(dest_path)
        # Only sync if explicitly requested (use with extreme caution)
        if force_sync: force_sync_directory(dest_path.parent)
    except Exception as e:
        success = False; error_msg = str(e)
        print(f"ERROR saving document {dest}: {error_msg}")
        if not disable_wandb:
            try: wandb.log({"document_save_errors": wandb.Table(data=[[dest, error_msg]], columns=["Path", "Error Message"])})
            except Exception as log_e: print(f"Error logging save error to W&B: {log_e}")
    finally:
        save_time = time.time() - start_time
        if not disable_wandb:
            try: wandb.log({"document_save_time": save_time, "document_save_success": success})
            except Exception as log_e: print(f"Error logging save metrics to W&B: {log_e}")

def split_text_on_full_stop(text: str, max_length: int) -> list:
    """Splits text smartly respecting sentence boundaries."""
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

def log_nlp_statistics(doc, step, doc_id=None, disable_wandb=False):
    """Log NLP statistics for a document to wandb"""
    # (Function unchanged)
    if disable_wandb: return
    try:
        token_count = len(doc)
        if token_count == 0: wandb.log({"nlp_stats_warnings": f"Document {doc_id} has 0 tokens"}, step=step); return
        unique_tokens = len(set([token.text.lower() for token in doc]))
        pos_counts = {}; ent_counts = {}
        for token in doc: pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        for ent in doc.ents: ent_counts[ent.label_] = ent_counts.get(ent.label_, 0) + 1
        try: sentences = list(doc.sents); sentence_count = len(sentences)
        except: sentences=[]; sentence_count=0 # Handle potential errors during sentencizing
        avg_sentence_length = token_count / sentence_count if sentence_count > 0 else 0
        token_lengths = [len(token.text) for token in doc]
        avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        has_sentiment = "sentiment" in doc.user_data
        sentiment = doc.user_data.get("sentiment", 0) if has_sentiment else None
        sentence_length_hist = None
        if sentence_count > 0:
            sentence_lengths = [len(sent) for sent in sentences]
            if sentence_lengths: # Ensure list is not empty
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

def log_system_metrics(disable_wandb=False):
    """Log system resource usage metrics"""
    # (Function unchanged)
    if disable_wandb: return
    try:
        cpu_percent = psutil.cpu_percent(interval=None); cpu_times = psutil.cpu_times_percent(interval=None)
        memory = psutil.virtual_memory(); swap = psutil.swap_memory()
        try: disk = psutil.disk_usage('/')
        except FileNotFoundError: disk = None
        net_stats = {}
        try:
            net_io = psutil.net_io_counters()
            net_stats = {"net_bytes_sent": net_io.bytes_sent, "net_bytes_recv": net_io.bytes_recv, "net_packets_sent": net_io.packets_sent, "net_packets_recv": net_io.packets_recv}
        except Exception: pass
        try: load_avg = os.getloadavg(); sys_load = {"system_load_1min": load_avg[0], "system_load_5min": load_avg[1], "system_load_15min": load_avg[2]}
        except AttributeError: sys_load = {}
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

def log_server_metrics(disable_wandb=False):
    """Enhanced logging for server environments"""
    # (Function unchanged)
    if disable_wandb or not numa: return
    try:
        server_log = {}; numa_stats = {}
        if numa:
            num_nodes = numa.num_configured_nodes()
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

def manage_memory():
    """Periodically clean up memory"""
    # (Function unchanged)
    try:
        collected = gc.collect(); print(f"Garbage Collector: Freed {collected} objects.")
        if torch.cuda.is_available(): torch.cuda.empty_cache(); print("Cleared CUDA cache.")
    except Exception as e: print(f"Error during memory management: {e}")


# --- Document Processing Function ---

def process_document(text: str, nlp: Language, dest: str, doc_id: str = None, disable_wandb: bool = False, force_sync: bool = False) -> Tuple[bool, str, float, Dict]:
    """Turns text into a spaCy document, using nlp.pipe for parallelism if text is split."""
    metrics = {"doc_length": len(text), "had_to_split": len(text) > MAX_LENGTH}
    start_time = time.time()

    try:
        # Optional: Track memory within this function
        # peak_memory_before = psutil.Process().memory_info().rss / (1024 * 1024)

        segment_size = MAX_LENGTH # Default segment size
        if len(text) > MAX_LENGTH:
            segment_size = MAX_LENGTH // 2 if SYSTEM_CONFIG["CPU_COUNT"] > 32 else MAX_LENGTH
            texts = split_text_on_full_stop(text, segment_size)
            metrics["num_segments"] = len(texts)
            metrics["avg_segment_length"] = sum(len(t) for t in texts) / len(texts) if texts else 0
            metrics["segment_size_used"] = segment_size
            if not disable_wandb:
                 try: wandb.log({"text_splitting": wandb.Table(data=[[doc_id, len(text), len(texts), MAX_LENGTH, segment_size]], columns=["Document ID", "Original Length", "Number of Segments", "Max Length", "Segment Size Used"])})
                 except Exception as log_e: print(f"Error logging splitting table: {log_e}")

            # --- Use nlp.pipe for parallel processing of segments ---
            pipe_start_time = time.time()
            docs = list(nlp.pipe(texts, n_process=N_PROCESS, batch_size=BATCH_SIZE))
            pipe_time = time.time() - pipe_start_time
            metrics["pipe_processing_time"] = pipe_time
            # -------------------------------------------------------

        else:
            # --- Process single text directly (or use pipe for consistency?) ---
            # Option 1: Direct processing (potentially less overhead for single short texts)
            # docs = [nlp(text)]
            # metrics["pipe_processing_time"] = 0 # Or time the direct call

            # Option 2: Use pipe even for one item (consistent, uses configured resources)
            pipe_start_time = time.time()
            docs = list(nlp.pipe([text], n_process=1, batch_size=1)) # Use n_process=1 for single item
            pipe_time = time.time() - pipe_start_time
            metrics["pipe_processing_time"] = pipe_time
            # -----------------------------------------------------------
            metrics["num_segments"] = 1
            metrics["avg_segment_length"] = len(text)


        # Combine the processed segments/doc back into a single document
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
        metrics["token_count"] = len(doc)
        metrics["entity_count"] = len(doc.ents)
        try: metrics["sentence_count"] = len(list(doc.sents)) if len(doc) > 0 else 0
        except Exception as sent_e:
            print(f"Warning: Could not count sentences for {doc_id}: {sent_e}")
            metrics["sentence_count"] = 0

        # Save document
        save_start_time = time.time()
        save_document(doc, dest=dest, disable_wandb=disable_wandb, force_sync=force_sync) # Pass force_sync flag
        save_time = time.time() - save_start_time
        metrics["doc_save_time"] = save_time

        processing_time = time.time() - start_time
        metrics["total_processing_time"] = processing_time

        # peak_memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
        # metrics["memory_usage_mb"] = peak_memory_after - peak_memory_before

        return True, "", processing_time, metrics
    except Exception as e:
        error_message = f"Error in process_document for {doc_id}: {str(e)}"
        print(error_message)
        if not disable_wandb:
             try: wandb.log({"processing_errors": wandb.Table(data=[[doc_id, dest, str(e)]], columns=["Document ID", "Document Path", "Error Message"])})
             except Exception as log_e: print(f"Error logging processing error: {log_e}")
        metrics["total_processing_time"] = time.time() - start_time
        return False, str(e), metrics.get("total_processing_time", 0.0), metrics

# --- Worker Function Removed ---
# (process_batch function is deleted)

# --- CLI Parser ---
# (create_parser unchanged, still includes --batch_items but it won't be used)
def create_parser() -> argparse.ArgumentParser:
    """Creates parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="Corpus processor",
        description="Processes all documents in a corpus on CPU using nlp.pipe.",
    )
    parser.add_argument("--model", type=str, default="grc_proiel_trf", help="Name of the spaCy model to use.")
    parser.add_argument("--dest", type=str, default="dat/greek/processed_data/", help="Destination directory for processed .spacy files.")
    parser.add_argument("--src_index", type=str, default="dat/greek/cleaned_parsed_data/index.csv", help="Path to the CSV index file containing source document paths.")
    parser.add_argument("--wandb_user", type=str, default="sozialismus-au", help="Weights & Biases username or entity.")
    parser.add_argument("--wandb_project", type=str, default="model-tracking", help="Weights & Biases project name.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional name for the W&B run.")
    parser.add_argument("--sample_interval", type=int, default=50, help="Log detailed system/server metrics every N documents.")
    parser.add_argument("--checkpoint_interval", type=int, default=200, help="Save checkpoint data every N documents.")
    # parser.add_argument("--batch_items", type=int, default=64, help="[DEPRECATED] Number of documents to group into a single batch for worker processing.") # Mark as deprecated
    parser.add_argument("--disable_wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--force_sync", action="store_true", help="Force disk sync after each file save (SLOW, use for critical data only).")
    return parser

# --- Main Execution ---

def main():
    parser = create_parser()
    args = parser.parse_args()
    print(
        "--------------------------\n"
        "-- PROCESS CORPUS (Single Model Load) --\n"
        "--------------------------\n"
    )

    # Creating destination directory
    dest_path_obj = Path(args.dest); dest_path_obj.mkdir(exist_ok=True, parents=True)
    print(f"Destination directory: {args.dest}")

    # Generate run name
    run_name = args.run_name or f"corpus-processing-{args.model}-{time.strftime('%Y%m%d-%H%M%S')}"

    # --- Torch Thread Setup (Global) ---
    print(f"Setting global PyTorch threads to: {SYSTEM_CONFIG['TORCH_THREADS']}")
    torch.set_num_threads(SYSTEM_CONFIG['TORCH_THREADS'])

    # Initialize wandb
    if not args.disable_wandb:
        print(f"Initializing wandb with run name: {run_name}")
        try:
            wandb.init(project=args.wandb_project, entity=args.wandb_user, name=run_name, tags=["corpus-processing", args.model, "single-model"])
            wandb.config.update(vars(args))
            wandb.config.update({"system_config": SYSTEM_CONFIG, "max_length": MAX_LENGTH, "n_process_pipe": N_PROCESS, "batch_size_pipe": BATCH_SIZE, "device": "cpu"})
        except Exception as wandb_e:
             print(f"FATAL: Failed to initialize WandB: {wandb_e}"); args.disable_wandb = True; print("WandB logging has been disabled.")
    else: print("Weights & Biases logging disabled")

    # --- NUMA Setup ---
    numa_enabled_runtime = False
    if SYSTEM_CONFIG.get("NUMA_NODES", 1) > 1 and numa:
        numa_enabled_runtime = set_numa_awareness()
        print(f"NUMA awareness runtime status: {'Enabled' if numa_enabled_runtime else 'Not available/active'}")
        if not args.disable_wandb: wandb.config.update({"numa_enabled_runtime": numa_enabled_runtime})

    # --- Model Loading (Main Process) ---
    print(f"Loading NLP model: {args.model}...")
    nlp = None # Initialize nlp
    start_time_model_load = time.time()
    try:
        nlp = spacy.load(args.model)
        nlp.max_length = MAX_LENGTH # Set max length on the single loaded model
        model_load_time = time.time() - start_time_model_load
        print(f"Model loaded successfully in {model_load_time:.2f} seconds")
        mem_after_load = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"Memory after model load: {mem_after_load:.2f} MB")

        if not args.disable_wandb:
            wandb.log({"model_load_time_seconds": model_load_time, "memory_after_load_mb": mem_after_load})
            pipeline_components = [{"name": name, "type": str(type(component))} for name, component in nlp.pipeline]
            wandb.log({"model_pipeline": wandb.Table(data=[[comp["name"], comp["type"]] for comp in pipeline_components], columns=["Component", "Type"])})
    except Exception as e:
        error_message = f"Failed to load model '{args.model}': {str(e)}"
        print(f"CRITICAL ERROR: {error_message}")
        if not args.disable_wandb: wandb.log({"critical_errors": wandb.Table(data=[["Model Loading", error_message]], columns=["Stage", "Error"])})
        if wandb.run: wandb.finish() # Ensure wandb finishes if run started
        return # Cannot proceed without the model

    # --- Index Loading ---
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
        if not args.disable_wandb: wandb.log({"critical_errors": wandb.Table(data=[["Index Loading", error_message]], columns=["Stage", "Error"])})
        if wandb.run: wandb.finish()
        return

    # --- Filter Processed Files ---
    print(f"Checking for already processed files in {args.dest}...")
    done_ids = get_done_ids(args.dest)
    n_done = len(done_ids)
    if n_done > 0:
        done_filter = parsed_index.document_id.isin(done_ids)
        n_already_done_in_index = done_filter.sum()
        print(f"Found {n_done} existing .spacy files. Ignoring {n_already_done_in_index} matching entries in index.")
        parsed_index = parsed_index[~done_filter]
    else: print("No previously processed files found matching the index.")
    n_left = len(parsed_index.index)
    if n_left == 0:
        print("No documents left to process.");
        if not args.disable_wandb and wandb.run: wandb.finish()
        return
    print(f"Documents to process: {n_left}")
    if not args.disable_wandb:
        wandb.log({"documents_to_process": n_left, "documents_already_processed": n_done, "initial_progress": Plotly(progress_piechart(n_done, n_left + n_done))})

    # --- Prepare Iteration Data ---
    print("Preparing data for iteration...")
    src_paths = parsed_index['dest_path'].tolist() # Get paths as list
    doc_ids = parsed_index['document_id'].tolist() # Get ids as list
    doc_filenames = [str(dest_path_obj / f"{doc_id}.spacy") for doc_id in doc_ids] # Generate dest paths

    # --- Initialize Tracking ---
    start_time_total = time.time()
    processed_chars_total = 0
    successful_docs = 0
    failed_docs = 0
    total_processed_in_loop = 0
    checkpoint_dir = dest_path_obj / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_df = pd.DataFrame(columns=["document_id", "status", "processing_time", "timestamp"])

    # --- Main Processing Loop (Sequential Iteration) ---
    print(f"Starting processing {n_left} documents sequentially...")
    # Use stream_files to read text lazily
    texts_stream = stream_files(src_paths)

    for doc_out_path, text, doc_id in tqdm(zip(doc_filenames, texts_stream, doc_ids), total=n_left, desc="Processing Documents"):
        doc_index_in_run = total_processed_in_loop
        current_doc_global_index = n_done + doc_index_in_run

        # Process the document using the single loaded nlp object
        success, error_message, processing_time, metrics = process_document(
            text, nlp=nlp, dest=doc_out_path, doc_id=doc_id,
            disable_wandb=args.disable_wandb, force_sync=args.force_sync
        )

        # --- Update Counters & Checkpointing (Same as before) ---
        total_processed_in_loop += 1
        doc_length = metrics.get("doc_length", 0)
        if doc_length > 0: processed_chars_total += doc_length

        if success: successful_docs += 1; status = "success"
        else: failed_docs += 1; status = "failed"
        if not success and not args.disable_wandb:
             try: wandb.log({"failed_documents": wandb.Table(data=[[doc_id, error_message]], columns=["Document ID", "Error"])})
             except Exception as log_e: print(f"Error logging failed doc table: {log_e}")

        new_row = pd.DataFrame([{ "document_id": doc_id, "status": status, "processing_time": processing_time, "timestamp": time.time() }])
        checkpoint_df = pd.concat([checkpoint_df, new_row], ignore_index=True)

        # --- Logging (Same as before) ---
        if not args.disable_wandb:
            elapsed_time = time.time() - start_time_total
            docs_per_sec = total_processed_in_loop / elapsed_time if elapsed_time > 0 else 0
            chars_per_sec = processed_chars_total / elapsed_time if elapsed_time > 0 else 0
            est_rem_time = (elapsed_time / total_processed_in_loop) * (n_left - total_processed_in_loop) if total_processed_in_loop > 0 else 0
            log_data = {
                "n_processed_total": current_doc_global_index, "n_processed_this_run": total_processed_in_loop,
                "progress_percent": (current_doc_global_index / (n_left + n_done)) * 100 if (n_left + n_done) > 0 else 0,
                "doc_length": doc_length, "processing_time_seconds": processing_time,
                "processing_speed_chars_per_sec": doc_length / processing_time if processing_time > 0 else 0,
                "cumulative_successful_docs": successful_docs, "cumulative_failed_docs": failed_docs,
                "elapsed_time_minutes": elapsed_time / 60, "avg_docs_per_second": docs_per_sec,
                "avg_chars_per_second": chars_per_sec, "estimated_remaining_time_minutes": est_rem_time / 60,
                **metrics }
            if total_processed_in_loop % 100 == 0 or total_processed_in_loop == n_left:
                 log_data["progress_pie"] = Plotly(progress_piechart(current_doc_global_index, n_left + n_done))
            try: wandb.log(log_data)
            except Exception as log_e: print(f"Error logging W&B data: {log_e}")

        # --- Periodic Tasks (Same as before) ---
        if total_processed_in_loop % args.checkpoint_interval == 0 and total_processed_in_loop > 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{current_doc_global_index}.csv"
            try:
                checkpoint_df.to_csv(checkpoint_path, index=False)
                print(f"\nSaved checkpoint ({len(checkpoint_df)} rows) at {checkpoint_path}")
                if not args.disable_wandb:
                    checkpoint_artifact = wandb.Artifact(f"checkpoint_{current_doc_global_index}", type="checkpoint")
                    checkpoint_artifact.add_file(str(checkpoint_path)); wandb.log_artifact(checkpoint_artifact)
            except Exception as cp_e: print(f"\nError saving checkpoint: {cp_e}")

        if total_processed_in_loop % args.sample_interval == 0 and total_processed_in_loop > 0:
            print(f"\nLogging system/server metrics at doc {total_processed_in_loop}...")
            log_system_metrics(args.disable_wandb)
            log_server_metrics(args.disable_wandb)
            # Log NLP stats for the *last successfully processed* document
            if success:
                 try:
                     # Need the actual Doc object, which we have from process_document return if we modify it,
                     # or re-process sample text here. Let's reprocess for simplicity now.
                     print(f"Logging NLP stats for sample of doc {doc_id}...")
                     sample_text = text[:min(len(text), MAX_LENGTH)]
                     # Need to handle potential errors here if text is empty or very short
                     if sample_text:
                         sample_doc = nlp(sample_text) # Use the main nlp object
                         log_nlp_statistics(sample_doc, current_doc_global_index, doc_id, args.disable_wandb)
                     else:
                         print(f"Skipping NLP stats for empty/short sample of doc {doc_id}")
                 except Exception as e:
                      print(f"\nError logging NLP stats: {e}")
                      if not args.disable_wandb: wandb.log({"statistics_errors": str(e)})


        if total_processed_in_loop % 100 == 0 and total_processed_in_loop > 0:
            print("\nRunning periodic memory management...")
            manage_memory()

    print("\nProcessing loop finished.")

    # --- Final Index and Checkpoint ---
    # (Logic unchanged)
    print("Creating final index for processed documents...")
    final_run_results = checkpoint_df[['document_id', 'status']].copy()
    # Need original index mapping doc_id to src_path ('dest_path' column in original csv)
    id_to_src_map = pd.Series(parsed_index.dest_path.values, index=parsed_index.document_id).to_dict()
    final_index = final_run_results.copy()
    final_index['src_path'] = final_index['document_id'].map(id_to_src_map)
    final_index['dest_path'] = final_index['document_id'].map(lambda did: str(dest_path_obj / f"{did}.spacy"))
    final_index.rename(columns={'status': 'processing_status'}, inplace=True)
    final_index = final_index[['document_id', 'src_path', 'dest_path', 'processing_status']] # Reorder

    index_path = dest_path_obj / "index_processed_run.csv"
    try:
        final_index.to_csv(index_path, index=False)
        print(f"Saved index for this run ({len(final_index)} entries) to {index_path}")
        if not args.disable_wandb:
            index_artifact = wandb.Artifact(f"processed_corpus_index_{run_name}", type="dataset_index")
            index_artifact.add_file(str(index_path)); wandb.log_artifact(index_artifact)
    except Exception as idx_e: print(f"Error saving final index: {idx_e}")

    final_checkpoint_path = checkpoint_dir / "final_checkpoint_run.csv"
    try:
        checkpoint_df.to_csv(final_checkpoint_path, index=False)
        print(f"Saved final run checkpoint ({len(checkpoint_df)} rows) to {final_checkpoint_path}")
        if not args.disable_wandb:
            final_checkpoint_artifact = wandb.Artifact(f"final_checkpoint_{run_name}", type="checkpoint")
            final_checkpoint_artifact.add_file(str(final_checkpoint_path)); wandb.log_artifact(final_checkpoint_artifact)
    except Exception as fcp_e: print(f"Error saving final checkpoint: {fcp_e}")

    # --- Summary ---
    # (Logic unchanged, including W&B float fix)
    processing_time_total = time.time() - start_time_total
    actual_processed_count = successful_docs + failed_docs
    success_rate_float = (successful_docs / actual_processed_count * 100) if actual_processed_count > 0 else 0.0
    success_rate_str = f"{success_rate_float:.2f}%"
    summary_data = [
        ["Total Documents in Index", n_total_in_index],
        ["Documents Already Processed (on disk)", n_done],
        ["Documents Attempted in This Run", actual_processed_count],
        ["Successfully Processed (this run)", successful_docs],
        ["Failed (this run)", failed_docs],
        ["Success Rate (this run)", success_rate_float], # Float for W&B
        ["Total Processing Time (min)", processing_time_total / 60],
        ["Avg Time per Document (sec, this run)", (processing_time_total / actual_processed_count) if actual_processed_count > 0 else 0.0],
        ["Avg Processing Speed (chars/sec, this run)", (processed_chars_total / processing_time_total) if processing_time_total > 0 else 0.0],
        ["Total Characters Processed (this run)", processed_chars_total],
        # Remove batch specific lines
        ["spaCy Processes (nlp.pipe)", N_PROCESS],
        ["spaCy Batch Size (nlp.pipe)", BATCH_SIZE],
        ["Max Document Length (chars)", MAX_LENGTH],
    ]
    print("\n===== Processing Summary =====")
    for name, value in summary_data:
        if name == "Success Rate (this run)": print(f"{name}: {success_rate_str}")
        elif isinstance(value, float): print(f"{name}: {value:.2f}")
        else: print(f"{name}: {value}")
    if not args.disable_wandb:
        try:
            summary_table_data = [[item[0], item[1]] for item in summary_data]
            wandb.log({"final_summary": wandb.Table(data=summary_table_data, columns=["Metric", "Value"])})
            if wandb.run: wandb.finish(); print("\nWandB run finished.")
        except Exception as log_e: print(f"Error logging final summary / finishing WandB: {log_e}")

if __name__ == "__main__":
    main()
