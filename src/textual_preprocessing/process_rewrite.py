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
import logging
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

# Import helper - exit if not found
try:
    from utils.streams import stream_files
except ImportError:
    logging.error("Could not import stream_files from utils.streams.")
    logging.error("Ensure utils/streams.py is accessible (e.g., in the same directory or PYTHONPATH).")
    sys.exit(1)

from wandb.data_types import Plotly

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Configuration ---

@dataclass
class RunConfig:
    """Configuration for the processing run."""
    # Paths
    src_index: Path # = "dat/greek/cleaned_parsed_data/index.csv" # Make required by argsparse
    dest_dir: Path # = "dat/greek/processed_data/" # Make required by argsparse
    # Model
    model_name: str
    # Performance Tuning
    max_length: int = 10_000 # Default from user's version
    n_process: int = max(1, min(16, (os.cpu_count() or 1) // 2)) # Default from user's version
    batch_size: int = 64 # Default from user's version
    torch_threads: int = max(1, min(8, (os.cpu_count() or 1) // 2)) # Default from user's version
    # Execution Control
    force_sync: bool = False
    sample_interval: int = 100
    checkpoint_interval: int = 500
    test_doc_id: Optional[str] = None # For test mode
    test_src_path: Optional[Path] = None # For test mode (Path object)
    # W&B
    wandb_user: Optional[str] = None
    wandb_project: Optional[str] = "model-tracking" # Default project from user's version
    run_name: Optional[str] = None
    disable_wandb: bool = False
    # Internal
    _run_id: str = field(default_factory=lambda: f"{time.strftime('%Y%m%d-%H%M%S')}")

    def __post_init__(self):
        # Ensure paths are Path objects (already done by argparse type=Path)
        # Generate run name if needed
        if not self.run_name:
            self.run_name = f"corpus-proc-{self.model_name}-{self._run_id}"

    def log_to_wandb(self):
        """Log configuration parameters to W&B."""
        if not self.disable_wandb and wandb.run:
            config_dict = self.__dict__.copy()
            for key, value in config_dict.items():
                if isinstance(value, Path):
                    config_dict[key] = str(value) # Convert Path to string for W&B
            config_dict["system/cpu_count"] = os.cpu_count()
            config_dict["system/memory_total_gb"] = psutil.virtual_memory().total / (1024**3)
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
#  split_text_on_full_stop, log_system_metrics, log_nlp_statistics,
#  manage_memory functions remain unchanged from previous provided version)
# ... (paste unchanged helper functions here) ...
def get_done_ids(path: Path) -> List[str]:
    """Finds documents that have already been cleaned using pathlib."""
    ids = []
    if path.is_dir():
        ids = [p.stem for p in path.glob("*.spacy") if p.is_file()]
    logging.info(f"Found {len(ids)} existing '.spacy' files in '{path}'.")
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
        title=f"Processing Progress: {n_processed}/{n_total}",
        height=300, width=400,
        margin=dict(l=20, r=20, t=40, b=20))
    return fig

def force_sync_directory(directory_path: Path):
    """Force system to sync directory to disk (use RARELY)."""
    logging.warning("Forcing disk sync - this significantly impacts performance!")
    try:
        if not directory_path.is_dir(): return
        if hasattr(os, 'sync'): os.sync()
        elif hasattr(os, 'fsync'):
            try:
                fd = os.open(str(directory_path), os.O_RDONLY)
                os.fsync(fd); os.close(fd)
            except OSError as e: logging.warning(f"Could not fsync dir {directory_path}: {e}")
        elif os.name == 'nt':
            import ctypes
            ctypes.windll.kernel32.FlushFileBuffers(ctypes.c_void_p(-1))
    except Exception as e: logging.warning(f"Could not force sync to disk: {e}")

def save_document(doc: Doc, dest_path: Path, config: RunConfig) -> Dict[str, Any]:
    """Serializes and saves spaCy Document using DocBin with robustness."""
    metrics = {"save_success": False, "save_time_sec": 0.0, "file_size_kb": 0.0}
    start_time = time.time()
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dest = dest_path.with_suffix(dest_path.suffix + ".tmp")
        if temp_dest.exists(): os.remove(temp_dest)
        doc_bin = DocBin(attrs=TOKEN_ATTRS, docs=[doc], store_user_data=True)
        doc_bin.to_disk(str(temp_dest))
        temp_dest.rename(dest_path)
        metrics["save_success"] = True
        metrics["file_size_kb"] = dest_path.stat().st_size / 1024
        if config.force_sync: force_sync_directory(dest_path.parent)
    except Exception as e:
        error_msg = f"Failed to save DocBin to {dest_path}: {e}"
        logging.error(error_msg)
        metrics["error"] = error_msg
        if temp_dest.exists():
            try: os.remove(temp_dest)
            except OSError: pass
        if not config.disable_wandb and wandb.run:
            try:
                wandb.log({"errors/document_save": wandb.Table(data=[[str(dest_path), str(e)]], columns=["Path", "Error Message"])})
            except Exception as log_e: logging.warning(f"Failed to log save error to W&B: {log_e}")
    finally:
        metrics["save_time_sec"] = time.time() - start_time
    return metrics

def split_text_on_full_stop(text: str, max_length: int) -> list:
    """Splits text smartly respecting sentence boundaries."""
    segments = []; start = 0; text_length = len(text)
    while start < text_length:
        if text_length - start <= max_length: segments.append(text[start:].strip()); break
        slice_end = start + max_length; segment = text[start:slice_end]
        split_index = segment.rfind('.')
        if split_index != -1: end = start + split_index + 1
        else:
            newline_index = segment.rfind('\n')
            if newline_index != -1: end = start + newline_index + 1
            else: end = slice_end
        segments.append(text[start:end].strip()); start = end
    return [seg for seg in segments if seg]

def log_system_metrics(config: RunConfig, step: int):
    """Log system resource usage metrics"""
    if config.disable_wandb or not wandb.run: return
    log_payload = {}
    try:
        log_payload["sys/cpu_util_percent"] = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        log_payload["sys/mem_util_percent"] = mem.percent
        log_payload["sys/mem_available_gb"] = mem.available / (1024 ** 3)
        process = psutil.Process()
        proc_mem = process.memory_info()
        log_payload["sys/proc_mem_rss_mb"] = proc_mem.rss / (1024 * 1024)
        log_payload["sys/proc_cpu_util_percent"] = process.cpu_percent(interval=None)
        log_payload["sys/proc_threads"] = process.num_threads()
        try:
            disk = psutil.disk_usage(str(config.dest_dir))
            log_payload["sys/disk_util_percent"] = disk.percent
            log_payload["sys/disk_free_gb"] = disk.free / (1024 ** 3)
        except FileNotFoundError: pass
        try:
            load_avg = os.getloadavg()
            log_payload["sys/load_avg_1min"] = load_avg[0]
        except AttributeError: pass
        wandb.log(log_payload, step=step)
    except Exception as e: logging.warning(f"Could not log system metrics: {e}")

def log_nlp_statistics(doc: Doc, step: int, doc_id: Optional[str], config: RunConfig):
    """Log NLP statistics for a document to wandb"""
    if config.disable_wandb or not wandb.run: return
    stats_log = {}
    try:
        token_count = len(doc)
        stats_log["nlp/token_count"] = token_count
        if token_count == 0: return
        stats_log["nlp/unique_tokens"] = len(set([token.text.lower() for token in doc]))
        stats_log["nlp/lexical_diversity"] = stats_log["nlp/unique_tokens"] / token_count if token_count > 0 else 0
        pos_counts = {}
        for token in doc: pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        if pos_counts:
             pos_data = [[pos, count, f"{(count/token_count*100):.1f}%"] for pos, count in sorted(pos_counts.items(), key=lambda item: item[1], reverse=True)]
             stats_log["nlp/pos_distribution"] = wandb.Table(data=pos_data, columns=["POS", "Count", "Percentage"])
        stats_log["nlp/entity_count"] = len(doc.ents)
        ent_counts = {}
        for ent in doc.ents: ent_counts[ent.label_] = ent_counts.get(ent.label_, 0) + 1
        for ent_type, count in ent_counts.items(): stats_log[f"nlp/entities/{ent_type}"] = count
        try: sentences = list(doc.sents); sentence_count = len(sentences)
        except: sentences=[]; sentence_count=0
        stats_log["nlp/sentence_count"] = sentence_count
        stats_log["nlp/avg_sentence_len_tokens"] = token_count / sentence_count if sentence_count > 0 else 0
        wandb.log(stats_log, step=step)
    except Exception as e: logging.warning(f"Could not log NLP stats for doc {doc_id}: {e}")

def manage_memory(step: int):
    """Periodically run garbage collection."""
    try:
        start_mem = psutil.Process().memory_info().rss / (1024*1024)
        collected = gc.collect()
        end_mem = psutil.Process().memory_info().rss / (1024*1024)
        logging.info(f"[{step}] Garbage Collection: Freed {collected} objects. Mem before: {start_mem:.1f}MB, after: {end_mem:.1f}MB")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e: logging.warning(f"Error during GC at step {step}: {e}")

# --- Core Processing Function ---

def process_document(text: str, doc_id: str, dest_path: Path, nlp: Language, config: RunConfig) -> Tuple[bool, Dict[str, Any], Optional[Doc]]:
    """
    Processes a single document text, handles splitting, uses nlp.pipe, saves result.
    Returns tuple: (success_status, metrics_dict, final_doc_object)
    """
    # <<< DEBUG PRINT ADDED >>>
    logging.debug(f"[DEBUG {doc_id}] Entering process_document. Text length: {len(text)}")
    process_start_time = time.time()
    metrics = {
        "doc_id": doc_id, "doc_length": len(text), "had_to_split": False, "num_segments": 1,
        "total_proc_time_sec": 0.0, "split_time_sec": 0.0, "pipe_time_sec": 0.0,
        "combine_time_sec": 0.0, "save_time_sec": 0.0, "token_count": 0,
        "error": None
    }
    success = False
    final_doc: Optional[Doc] = None

    try:
        # 1. Splitting (if necessary)
        split_start_time = time.time()
        # <<< DEBUG PRINT ADDED >>>
        logging.debug(f"[DEBUG {doc_id}] Checking if splitting needed (max_length={config.max_length})...")
        texts = [text]
        if len(text) > config.max_length:
            logging.info(f"Doc '{doc_id}' length {len(text)} > max_length {config.max_length}. Splitting...")
            metrics["had_to_split"] = True
            texts = split_text_on_full_stop(text, config.max_length)
            metrics["num_segments"] = len(texts)
            # <<< DEBUG PRINT ADDED >>>
            logging.debug(f"[DEBUG {doc_id}] Splitting done. Segments: {metrics['num_segments']}")
            if not texts:
                logging.warning(f"Splitting doc '{doc_id}' resulted in zero segments. Creating empty doc.")
                texts = []
        else:
             # <<< DEBUG PRINT ADDED >>>
            logging.debug(f"[DEBUG {doc_id}] No splitting needed.")
        metrics["split_time_sec"] = time.time() - split_start_time

        # 2. spaCy Processing (using nlp.pipe)
        pipe_start_time = time.time()
        docs: List[Doc] = []
        if texts:
            non_empty_texts = [t for t in texts if t]
            if non_empty_texts:
                 current_n_process = 1 if len(non_empty_texts) == 1 else config.n_process
                 current_batch_size = config.batch_size # Use configured batch size regardless
                 # <<< DEBUG PRINT ADDED >>>
                 logging.debug(f"[DEBUG {doc_id}] Calling nlp.pipe with n_process={current_n_process}, batch_size={current_batch_size} for {len(non_empty_texts)} segments...")
                 try:
                     processed_docs = nlp.pipe(non_empty_texts, n_process=current_n_process, batch_size=current_batch_size)
                     docs = list(processed_docs)
                     # <<< DEBUG PRINT ADDED >>>
                     logging.debug(f"[DEBUG {doc_id}] nlp.pipe finished.")
                 except Exception as pipe_e:
                     logging.error(f"nlp.pipe failed for doc '{doc_id}': {pipe_e}")
                     metrics["error"] = f"nlp.pipe error: {pipe_e}"
                     docs = []
            else:
                 logging.warning(f"No non-empty text segments found for doc {doc_id} after splitting.")
        metrics["pipe_time_sec"] = time.time() - pipe_start_time

        # 3. Combine Docs (if split or processed)
        # <<< DEBUG PRINT ADDED >>>
        logging.debug(f"[DEBUG {doc_id}] Combining {len(docs)} processed docs...")
        combine_start_time = time.time()
        if len(docs) > 1:
            final_doc = Doc.from_docs(docs)
        elif len(docs) == 1:
            final_doc = docs[0]
        else:
            logging.warning(f"Creating empty Doc object for '{doc_id}'.")
            final_doc = Doc(nlp.vocab)
        # <<< DEBUG PRINT ADDED >>>
        logging.debug(f"[DEBUG {doc_id}] Combining finished.")
        metrics["combine_time_sec"] = time.time() - combine_start_time

        # 4. Get Basic Doc Stats (moved after potential combine)
        metrics["token_count"] = len(final_doc) if final_doc else 0

        # 5. Save Document
        # <<< DEBUG PRINT ADDED >>>
        logging.debug(f"[DEBUG {doc_id}] Saving document...")
        save_metrics = save_document(final_doc, dest_path, config)
        metrics.update(save_metrics)
        # <<< DEBUG PRINT ADDED >>>
        logging.debug(f"[DEBUG {doc_id}] Saving finished. Success: {metrics['save_success']}")

        success = metrics["save_success"] and metrics["error"] is None

    except Exception as e:
        logging.exception(f"Unexpected critical error processing doc '{doc_id}': {e}")
        metrics["error"] = f"Critical error: {e}"
        success = False
        if not config.disable_wandb and wandb.run:
             try: wandb.log({"errors/critical_processing": wandb.Table(data=[[doc_id, str(dest_path), str(e)]], columns=["Doc ID", "Path", "Error"])})
             except Exception as log_e: logging.warning(f"Failed to log critical error to W&B: {log_e}")

    finally:
        metrics["total_proc_time_sec"] = time.time() - process_start_time
        # <<< DEBUG PRINT ADDED >>>
        logging.debug(f"[DEBUG {doc_id}] Exiting process_document. Total time: {metrics['total_proc_time_sec']:.2f}s")

    return success, metrics, final_doc


# --- CLI Parser ---

def create_parser() -> argparse.ArgumentParser:
    """Creates parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="Corpus processor",
        description="Processes documents using spaCy (single model load, nlp.pipe parallelism).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Required Paths (use type=Path for automatic conversion)
    parser.add_argument("--src_index", type=Path, required=True, help="Path to the source CSV index file (must contain 'document_id', 'dest_path').")
    parser.add_argument("--dest_dir", type=Path, required=True, help="Destination directory for processed .spacy files.")
    # Model
    parser.add_argument("--model", type=str, default="en_core_web_sm", dest="model_name", help="Name of the spaCy model to use (e.g., en_core_web_trf, grc_proiel_trf).") # Use dest for RunConfig name
    # Performance Tuning
    parser.add_argument("--max_length", type=int, help=f"Max document length (chars) before splitting. Overrides default ({RunConfig.max_length}).")
    parser.add_argument("--n_process", type=int, help=f"Processes for nlp.pipe parallelism. Overrides default ({RunConfig.n_process}). TUNABLE.")
    parser.add_argument("--batch_size", type=int, help=f"Batch size for nlp.pipe. Overrides default ({RunConfig.batch_size}). TUNABLE.")
    parser.add_argument("--torch_threads", type=int, help=f"Max PyTorch threads for main process. Overrides default ({RunConfig.torch_threads}).")
    # Execution Control
    parser.add_argument("--force_sync", action="store_true", help="Force disk sync after saves (EXTREMELY SLOW).")
    parser.add_argument("--sample_interval", type=int, help=f"Log detailed system/NLP metrics every N docs. Overrides default ({RunConfig.sample_interval}).")
    parser.add_argument("--checkpoint_interval", type=int, help=f"Save checkpoint data every N docs. Overrides default ({RunConfig.checkpoint_interval}).")
    # --- Test Mode Arguments ---
    parser.add_argument("--test_doc_id", type=str, default=None, help="If set, process only this document ID and exit.")
    parser.add_argument("--test_src_path", type=Path, default=None, help="Optional: Direct path to source text file for test mode.") # Use type=Path
    # --------------------------
    # W&B
    parser.add_argument("--wandb_user", type=str, default="sozialismus-au", help="W&B user/entity (or WANDB_ENTITY env var).")
    parser.add_argument("--wandb_project", type=str, default="model-tracking", help="W&B project (or WANDB_PROJECT env var).")
    parser.add_argument("--run_name", type=str, default=None, help="Optional W&B run name (default: auto-generated).")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B logging.")

    return parser

# --- Main Execution ---

def main():
    """Main function to orchestrate corpus processing."""
    parser = create_parser()
    args = parser.parse_args()

    # --- Initialize Configuration ---
    # Collect args that were *actually* provided by the user to override defaults
    cli_args = {k: v for k, v in vars(args).items() if v is not None}

    try:
        # Initialize config using defaults or required args, then apply CLI overrides
        config = RunConfig(
            src_index=args.src_index, # Required
            dest_dir=args.dest_dir,   # Required
            model_name=args.model_name, # Required (mapped from --model)
            # Apply other args if they were set via CLI
            max_length=args.max_length if args.max_length is not None else RunConfig.max_length,
            n_process=args.n_process if args.n_process is not None else RunConfig.n_process,
            batch_size=args.batch_size if args.batch_size is not None else RunConfig.batch_size,
            torch_threads=args.torch_threads if args.torch_threads is not None else RunConfig.torch_threads,
            force_sync=args.force_sync,
            sample_interval=args.sample_interval if args.sample_interval is not None else RunConfig.sample_interval,
            checkpoint_interval=args.checkpoint_interval if args.checkpoint_interval is not None else RunConfig.checkpoint_interval,
            test_doc_id=args.test_doc_id,
            test_src_path=args.test_src_path,
            wandb_user=args.wandb_user,
            wandb_project=args.wandb_project,
            run_name=args.run_name,
            disable_wandb=args.disable_wandb,
        )
    except Exception as config_e:
        logging.exception(f"Error initializing configuration: {config_e}")
        sys.exit(1)

    # Setup logging level for DEBUG prints if needed (e.g., during test mode)
    if config.test_doc_id:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Test mode activated, setting log level to DEBUG.")

    logging.info(f"Starting corpus processing run: {config.run_name}")
    logging.info(f"Using Config: {config}") # Dataclass repr is useful

    # --- Directory Setup ---
    try:
        config.dest_dir.mkdir(exist_ok=True, parents=True)
        checkpoint_dir = config.dest_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        logging.info(f"Output directory: {config.dest_dir}")
        logging.info(f"Checkpoint directory: {checkpoint_dir}")
    except OSError as e:
        logging.exception(f"Could not create output/checkpoint directories: {e}")
        sys.exit(1)

    # --- Torch Thread Setup ---
    # >>> CRITICAL FIX - Set threads BEFORE model load <<<
    # Setting to 1 is the recommended workaround for PyTorch/multiprocessing deadlocks
    num_torch_threads_to_set = 1
    logging.info(f"Setting global PyTorch threads to: {num_torch_threads_to_set} (Workaround for multiprocessing issues)")
    torch.set_num_threads(num_torch_threads_to_set)
    # Update config to reflect the actual setting (though model was loaded with 1)
    config.torch_threads = num_torch_threads_to_set # Reflect the change in config display/logging

    # --- Initialize W&B ---
    if not config.disable_wandb:
        # Check credentials before initializing
        if not config.wandb_user or not config.wandb_project:
             logging.warning("WANDB_ENTITY/--wandb_user or WANDB_PROJECT/--wandb_project not set. Disabling W&B.")
             config.disable_wandb = True
        # Check if logged in (optional, prevents crash later)
        # elif wandb.login(key=os.getenv("WANDB_API_KEY")) is False: # Requires API key env var
        #      logging.warning("Not logged into W&B or API key invalid. Disabling W&B.")
        #      config.disable_wandb = True
        else:
            logging.info(f"Initializing W&B (User: {config.wandb_user}, Project: {config.wandb_project}, Run: {config.run_name})...")
            try:
                wandb.init(project=config.wandb_project, entity=config.wandb_user, name=config.run_name,
                           tags=["corpus-processing", config.model_name, "single-model"])
                config.log_to_wandb() # Log the final config values
            except Exception as wandb_e:
                 logging.error(f"Error initializing WandB: {wandb_e}. Disabling W&B for this run.")
                 config.disable_wandb = True
    else:
        logging.info("Weights & Biases logging explicitly disabled.")


    # --- Load spaCy Model ---
    logging.info(f"Loading spaCy model: {config.model_name}...")
    nlp: Optional[Language] = None
    start_time_model_load = time.time()
    try:
        with warnings.catch_warnings():
             warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
             nlp = spacy.load(config.model_name, exclude=[])
        nlp.max_length = config.max_length
        model_load_time = time.time() - start_time_model_load
        mem_after_load = psutil.Process().memory_info().rss / (1024 * 1024)
        logging.info(f"Model loaded successfully in {model_load_time:.2f} seconds.")
        logging.info(f"Memory after model load: {mem_after_load:.2f} MB")

        if not config.disable_wandb and wandb.run:
            wandb.log({"perf/model_load_time_sec": model_load_time, "mem/after_load_mb": mem_after_load})
            try:
                pipe_names = nlp.pipe_names
                wandb.summary["model/pipeline_components"] = pipe_names
            except Exception as pipe_e: logging.warning(f"Could not log pipeline components: {pipe_e}")

    except Exception as e:
        logging.exception(f"Failed to load model '{config.model_name}'. Exiting.")
        if not config.disable_wandb and wandb.run:
            wandb.log({"errors/critical": wandb.Table(data=[["Model Loading", str(e)]], columns=["Stage", "Error"])})
            if wandb.run: wandb.finish(exit_code=1)
        sys.exit(1)


    # --- Load Index (Needed for both modes unless test_src_path is given) ---
    logging.info(f"Loading index: {config.src_index}")
    parsed_index: Optional[pd.DataFrame] = None
    n_total_in_index = 0
    try:
        if not config.src_index.is_file():
             raise FileNotFoundError(f"Source index file not found: {config.src_index}")
        parsed_index = pd.read_csv(config.src_index)
        required_cols = ['document_id', 'dest_path']
        if not all(col in parsed_index.columns for col in required_cols):
             raise ValueError(f"Index CSV '{config.src_index}' must contain columns: {required_cols}")
        parsed_index['document_id'] = parsed_index['document_id'].astype(str)
        if parsed_index['document_id'].duplicated().any():
             logging.warning("Duplicate document IDs found in index. Processing duplicates.")
        n_total_in_index = len(parsed_index.index)
        logging.info(f"Loaded index with {n_total_in_index} documents.")
        if not config.disable_wandb and wandb.run:
            wandb.log({"data/total_documents_in_index": n_total_in_index})
    except Exception as e:
        # Allow continuing in test mode only if test_src_path is provided
        if config.test_doc_id and config.test_src_path:
             logging.warning(f"Failed to load index, but continuing in test mode with provided --test_src_path: {e}")
             parsed_index = None
        else:
            logging.exception(f"Failed to load or validate index '{config.src_index}'. Exiting.")
            if not config.disable_wandb and wandb.run:
                 wandb.log({"errors/critical": wandb.Table(data=[["Index Loading", str(e)]], columns=["Stage", "Error"])})
                 if wandb.run: wandb.finish(exit_code=1)
            sys.exit(1)

    # ==============================================================
    # ======= SINGLE FILE TEST BLOCK ===============================
    # ==============================================================
    if config.test_doc_id:
        logging.info("*** RUNNING SINGLE FILE TEST ***")
        TEST_DOC_ID = config.test_doc_id
        TEST_SRC_PATH = config.test_src_path # Already a Path object or None

        # Find source path from index if not provided directly
        if not TEST_SRC_PATH:
            if parsed_index is None:
                 logging.error("Index failed to load and --test_src_path not provided. Cannot find test file.")
                 if wandb.run: wandb.finish(exit_code=1)
                 sys.exit(1)
            logging.info(f"Looking up source path for ID '{TEST_DOC_ID}' in index...")
            doc_row = parsed_index[parsed_index['document_id'] == TEST_DOC_ID]
            if doc_row.empty:
                logging.error(f"Test Document ID '{TEST_DOC_ID}' not found in index '{config.src_index}'. Provide path via --test_src_path or use a valid ID.")
                if wandb.run: wandb.finish(exit_code=1)
                sys.exit(1)
            TEST_SRC_PATH = Path(doc_row.iloc[0]['dest_path'])
            logging.info(f"Found source path from index: {TEST_SRC_PATH}")

        TEST_DEST_PATH = config.dest_dir / f"{TEST_DOC_ID}.spacy"

        logging.info(f"Test Config:")
        logging.info(f"  Doc ID: {TEST_DOC_ID}")
        logging.info(f"  Source: {TEST_SRC_PATH}")
        logging.info(f"  Dest:   {TEST_DEST_PATH}")
        logging.info(f"  n_process: {config.n_process}, batch_size: {config.batch_size}, max_length: {config.max_length}")

        # Read the test file
        try:
            if not TEST_SRC_PATH.is_file():
                logging.error(f"Test source file not found: {TEST_SRC_PATH}"); sys.exit(1)
            with open(TEST_SRC_PATH, 'r', encoding='utf-8') as f:
                test_text = f.read()
            logging.info(f"Read {len(test_text):,} characters from test file.")
            if len(test_text) <= config.max_length:
                 logging.warning("Test file is NOT longer than max_length. nlp.pipe multiprocessing might not use multiple processes.")
        except Exception as e:
            logging.exception(f"Failed reading test file {TEST_SRC_PATH}"); sys.exit(1)

        # Call process_document for the test file
        logging.info(f"Starting processing for test file...")
        test_success, test_metrics, test_final_doc = process_document(
            test_text, doc_id=TEST_DOC_ID, dest_path=TEST_DEST_PATH, nlp=nlp, config=config
        )

        # Print Results
        print("\n" + "***" * 10 + " SINGLE FILE TEST COMPLETE " + "***" * 10)
        print(f"  Success: {test_success}")
        print(f"  Metrics:")
        for key, value in test_metrics.items():
            if isinstance(value, float): print(f"    {key:<30}: {value:.4f}")
            else: print(f"    {key:<30}: {value}")

        # Optionally log detailed NLP stats for the test doc
        if test_success and test_final_doc and not config.disable_wandb and wandb.run:
             logging.info("Logging NLP stats for test document to W&B...")
             log_nlp_statistics(test_final_doc, step=0, doc_id=TEST_DOC_ID, config=config) # Log at step 0

        # Exit cleanly
        logging.info("Exiting after single file test.")
        if not config.disable_wandb and wandb.run:
            # Log test metrics to W&B as a summary dict
            wandb.summary["test_mode_results"] = test_metrics
            wandb.summary["test_mode_success"] = test_success
            wandb.finish(exit_code=0)
        sys.exit(0)
    # ==============================================================
    # ===== END OF SINGLE FILE TEST BLOCK ==========================
    # ==============================================================


    # --- Filter Processed Files (Full Run) ---
    logging.info(f"Checking for already processed files in {config.dest_dir}...")
    done_ids = get_done_ids(config.dest_dir)
    n_done = len(done_ids)
    parsed_index_filtered = parsed_index # Start with the full loaded index
    if n_done > 0:
        done_filter = parsed_index['document_id'].isin(done_ids)
        n_already_done_in_index = done_filter.sum()
        logging.info(f"Ignoring {n_already_done_in_index} documents found in index that already have output files.")
        parsed_index_filtered = parsed_index[~done_filter].copy()
    else:
        logging.info("No previously processed files found.")

    n_left = len(parsed_index_filtered.index)
    if n_left == 0:
        logging.info("No documents left to process based on index and existing files.")
        if not config.disable_wandb and wandb.run: wandb.finish()
        sys.exit(0)

    logging.info(f"Documents to process in this run: {n_left}")
    if not config.disable_wandb and wandb.run:
        wandb.log({"data/documents_to_process": n_left,
                   "data/documents_already_processed": n_done,
                   "progress/initial_pie": Plotly(progress_piechart(n_done, n_left + n_done))})


    # --- Prepare Iteration Data (Full Run) ---
    logging.info("Preparing data for iteration...")
    try:
        src_paths = parsed_index_filtered['dest_path'].tolist()
        doc_ids = parsed_index_filtered['document_id'].tolist()
        doc_filenames = [config.dest_dir / f"{doc_id}.spacy" for doc_id in doc_ids]
        texts_stream = stream_files(src_paths)
        logging.info("Data prepared.")
    except KeyError as e:
        logging.exception(f"Missing required column ('dest_path' or 'document_id') in index file. Exiting.")
        if not config.disable_wandb and wandb.run: wandb.finish(exit_code=1); sys.exit(1)
    except Exception as e:
        logging.exception(f"Error preparing iteration data: {e}. Exiting.")
        if not config.disable_wandb and wandb.run: wandb.finish(exit_code=1); sys.exit(1)


    # --- Initialize Tracking (Full Run) ---
    start_time_total = time.time()
    processed_chars_total = 0
    successful_docs = 0
    failed_docs = 0
    total_processed_in_loop = 0
    all_metrics: List[Dict] = []
    last_successful_doc: Optional[Doc] = None


    # --- Main Processing Loop (Full Run) ---
    logging.info(f"Starting processing {n_left} documents...")
    pbar = tqdm(zip(doc_filenames, texts_stream, doc_ids), total=n_left, desc="Processing Docs", unit="doc")

    try:
        for doc_out_path, text, doc_id in pbar:
            loop_step_start = time.time()
            logging.info(f"Processing Document ID: {doc_id} ({total_processed_in_loop + 1}/{n_left})...")

            # Process the document
            success, metrics, final_doc = process_document(
                text, doc_id=doc_id, dest_path=doc_out_path, nlp=nlp, config=config
            )
            all_metrics.append(metrics)
            if success and final_doc: last_successful_doc = final_doc

            # Update Counters
            total_processed_in_loop += 1; doc_length = metrics.get("doc_length", 0)
            if doc_length > 0: processed_chars_total += doc_length
            if success: successful_docs += 1
            else: failed_docs += 1

            # Live Logging
            current_doc_global_index = n_done + total_processed_in_loop
            loop_step_time_sec = time.time() - loop_step_start
            if not config.disable_wandb and wandb.run:
                elapsed_time = time.time() - start_time_total
                docs_per_sec = total_processed_in_loop / elapsed_time if elapsed_time > 0 else 0.0
                chars_per_sec = processed_chars_total / elapsed_time if elapsed_time > 0 else 0.0
                est_rem_time_sec = (elapsed_time / total_processed_in_loop) * (n_left - total_processed_in_loop) if total_processed_in_loop > 0 else 0.0
                log_data = {
                    "progress/n_processed_total": current_doc_global_index, "progress/n_processed_this_run": total_processed_in_loop,
                    "progress/percent": (current_doc_global_index / (n_left + n_done)) * 100 if (n_left + n_done) > 0 else 0.0,
                    "progress/current_doc_id": doc_id, "perf/doc_length": metrics.get("doc_length", 0),
                    "perf/total_proc_time_sec": metrics.get("total_proc_time_sec", 0.0), "perf/save_time_sec": metrics.get("save_time_sec", 0.0),
                    "perf/pipe_time_sec": metrics.get("pipe_time_sec", 0.0), "perf/throughput_docs_sec": docs_per_sec,
                    "perf/throughput_chars_sec": chars_per_sec, "time/elapsed_minutes": elapsed_time / 60,
                    "time/estimated_remaining_minutes": est_rem_time_sec / 60, "stats/cumulative_successful": successful_docs,
                    "stats/cumulative_failed": failed_docs, "stats/token_count": metrics.get("token_count", 0),
                    "stats/file_size_kb": metrics.get("file_size_kb", 0.0), "loop_step_time_sec": loop_step_time_sec, }
                try: wandb.log(log_data, step=current_doc_global_index)
                except Exception as log_e: logging.warning(f"Failed to log to W&B: {log_e}")

            # Periodic Tasks
            if total_processed_in_loop % config.checkpoint_interval == 0 and total_processed_in_loop > 0:
                logging.info(f"[{total_processed_in_loop}] Saving checkpoint...")
                checkpoint_path = checkpoint_dir / f"checkpoint_{n_done + total_processed_in_loop}.csv"
                try:
                    chkpt_df = pd.DataFrame(all_metrics); chkpt_df['timestamp'] = time.time()
                    chkpt_df.to_csv(checkpoint_path, index=False)
                    logging.info(f"Checkpoint saved to {checkpoint_path}")
                    if not config.disable_wandb and wandb.run:
                        try:
                             chkpt_artifact = wandb.Artifact(f"checkpoint_{n_done + total_processed_in_loop}", type="checkpoint")
                             chkpt_artifact.add_file(str(checkpoint_path)); wandb.log_artifact(chkpt_artifact)
                        except Exception as log_e: logging.warning(f"Failed to log checkpoint artifact: {log_e}")
                except Exception as cp_e: logging.error(f"Failed to save checkpoint: {cp_e}")

            if total_processed_in_loop % config.sample_interval == 0 and total_processed_in_loop > 0:
                logging.info(f"[{total_processed_in_loop}] Logging detailed system/NLP metrics...")
                log_system_metrics(config, step=current_doc_global_index)
                if last_successful_doc: log_nlp_statistics(last_successful_doc, current_doc_global_index, doc_id, config)
                else: logging.info(f"[{total_processed_in_loop}] Skipping NLP stats log (no recent successful doc).")

            if total_processed_in_loop % (config.checkpoint_interval // 2) == 0 and total_processed_in_loop > 0:
                manage_memory(total_processed_in_loop)

            pbar.set_description(f"Last: {doc_id[:15]} ({metrics.get('total_proc_time_sec', 0.0):.2f}s)")

    except KeyboardInterrupt:
        logging.warning("\nKeyboardInterrupt received. Exiting loop gracefully...")
    finally:
        pbar.close()
        logging.info("Processing loop finished or interrupted.")

    # --- Final Index and Checkpoint ---
    # (Final save logic unchanged)
    logging.info("Saving final index and full metrics checkpoint...")
    if not all_metrics: logging.warning("No documents were processed in this run. Skipping final save.")
    else:
        try:
            final_df = pd.DataFrame(all_metrics)
            final_df['processing_status'] = final_df['error'].apply(lambda x: 'failed' if pd.notna(x) else 'success')
            if parsed_index is not None:
                id_to_src_map = pd.Series(parsed_index['dest_path'].values, index=parsed_index['document_id']).to_dict()
                final_df['src_path'] = final_df['doc_id'].map(id_to_src_map)
            else: final_df['src_path'] = 'N/A (Index Load Failed)'
            final_df['dest_path'] = final_df['doc_id'].map(lambda did: str(config.dest_dir / f"{did}.spacy"))
            index_cols = ['doc_id', 'src_path', 'dest_path', 'processing_status', 'doc_length', 'token_count', 'total_proc_time_sec']
            final_index = final_df.reindex(columns=index_cols, fill_value=None)
            index_path = config.dest_dir / f"index_{config._run_id}.csv"
            final_index.to_csv(index_path, index=False)
            logging.info(f"Saved final index ({len(final_index)} entries) to {index_path}")
            if not config.disable_wandb and wandb.run:
                 try:
                      index_artifact = wandb.Artifact(f"processed_index_{config.run_name}", type="dataset_index")
                      index_artifact.add_file(str(index_path)); wandb.log_artifact(index_artifact)
                 except Exception as log_e: logging.warning(f"Failed to log index artifact: {log_e}")
            final_checkpoint_path = checkpoint_dir / f"final_metrics_{config._run_id}.csv"
            final_df.to_csv(final_checkpoint_path, index=False)
            logging.info(f"Saved final metrics checkpoint ({len(final_df)} rows) to {final_checkpoint_path}")
            if not config.disable_wandb and wandb.run:
                try:
                    metrics_artifact = wandb.Artifact(f"final_metrics_{config.run_name}", type="run_metrics")
                    metrics_artifact.add_file(str(final_checkpoint_path)); wandb.log_artifact(metrics_artifact)
                except Exception as log_e: logging.warning(f"Failed to log metrics artifact: {log_e}")
        except Exception as final_e: logging.exception(f"Error during final index/checkpoint creation: {final_e}")


    # --- Final Summary ---
    # (Summary logic unchanged)
    logging.info("Calculating final summary...")
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
    print("\n" + "="*25 + " Processing Summary " + "="*25)
    for name, value in summary_data:
        if name == "Success Rate (this run)": print(f"{name:<45}: {value:.2f}%")
        elif isinstance(value, float): print(f"{name:<45}: {value:.2f}")
        else: print(f"{name:<45}: {value}")
    print("="*70)
    if not config.disable_wandb and wandb.run:
        logging.info("Logging final summary to W&B...")
        try:
            summary_table_data = [[item[0], item[1]] for item in summary_data]
            wandb.log({"final_summary": wandb.Table(data=summary_table_data, columns=["Metric", "Value"])})
            wandb.log({ "perf/avg_doc_time_sec": avg_proc_time_sec, "perf/avg_chars_per_sec": avg_chars_per_sec,
                "stats/success_rate_percent": success_rate_float, "time/total_runtime_min": processing_time_total_sec / 60,
                "stats/total_successful": successful_docs, "stats/total_failed": failed_docs })
            logging.info("Finishing W&B run..."); wandb.finish(); logging.info("W&B run finished.")
        except Exception as log_e: logging.error(f"Failed to log final summary or finish W&B: {log_e}")

    logging.info("Script finished.")


if __name__ == "__main__":
    main()
