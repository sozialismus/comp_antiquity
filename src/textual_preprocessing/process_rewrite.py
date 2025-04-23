# -*- coding: utf-8 -*-
"""
Script responsible for cleaning/processing a corpus using spaCy.

Loads a single spaCy model and processes documents sequentially.
Uses spaCy's nlp.pipe() for parallelism when processing segments
of documents that exceed the configured max_length.
Logs progress, system metrics, and NLP statistics to Weights & Biases.
Allows selection of token attributes to save via --attributes flag.
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
    src_index: Path
    dest_dir: Path
    # Model
    model_name: str
    # Performance Tuning
    max_length: int = 10_000
    n_process: int = max(1, min(16, (os.cpu_count() or 1) // 2))
    batch_size: int = 64
    torch_threads: int = max(1, min(8, (os.cpu_count() or 1) // 2))
    # Execution Control
    attributes_key: str = "full" # <<< Key to select attribute set
    force_sync: bool = False
    sample_interval: int = 100
    checkpoint_interval: int = 500
    test_doc_id: Optional[str] = None
    test_src_path: Optional[Path] = None
    # W&B
    wandb_user: Optional[str] = None
    wandb_project: Optional[str] = "model-tracking"
    run_name: Optional[str] = None
    disable_wandb: bool = False
    # Internal
    _run_id: str = field(default_factory=lambda: f"{time.strftime('%Y%m%d-%H%M%S')}")

    def __post_init__(self):
        # Generate run name including attributes key
        if not self.run_name:
            self.run_name = f"corpus-proc-{self.model_name}-{self.attributes_key}-{self._run_id}"

    def log_to_wandb(self):
        """Log configuration parameters to W&B."""
        if not self.disable_wandb and wandb.run:
            config_dict = self.__dict__.copy()
            for key, value in config_dict.items():
                if isinstance(value, Path): config_dict[key] = str(value)
            config_dict["system/cpu_count"] = os.cpu_count()
            config_dict["system/memory_total_gb"] = psutil.virtual_memory().total / (1024**3)
            wandb.config.update(config_dict)


# --- Constants: Token Attribute Sets ---

TOKEN_ATTRS_FULL = [
    "IS_ALPHA", "IS_ASCII", "IS_DIGIT", "IS_LOWER", "IS_PUNCT", "IS_SPACE", "IS_TITLE", "IS_UPPER", "LIKE_URL", "LIKE_NUM", "LIKE_EMAIL", "IS_STOP",
    "IS_QUOTE", "IS_BRACKET", "IS_LEFT_PUNCT", "IS_RIGHT_PUNCT", "IS_CURRENCY", "ID", "ORTH", "LOWER", "NORM", "SHAPE", "PREFIX", "SUFFIX", "LENGTH",
    "LEMMA", "POS", "TAG", "DEP", "ENT_IOB", "ENT_TYPE", "ENT_ID", "ENT_KB_ID", "HEAD", "SENT_START", "SPACY", "LANG", "MORPH", "IDX",
]

TOKEN_ATTRS_NER_ID = [ # NER + Token ID
    "ORTH", "ENT_IOB", "ENT_TYPE", "SENT_START", "SPACY", "ID",
]

TOKEN_ATTRS_NER_ID_IDX = [ # NER + Token ID + Char Offset
    "ORTH", "ENT_IOB", "ENT_TYPE", "SENT_START", "SPACY", "ID", "IDX",
]

# --- Global variable to hold the selected attributes ---
# This will be set in main() based on config
SELECTED_TOKEN_ATTRS: List[str] = TOKEN_ATTRS_FULL # Default to full


# --- Helper Functions ---

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

# --- Updated save_document to use global SELECTED_TOKEN_ATTRS ---
def save_document(doc: Doc, dest_path: Path, config: RunConfig) -> Dict[str, Any]:
    """Serializes and saves spaCy Document using DocBin with robustness, using SELECTED_TOKEN_ATTRS."""
    metrics = {"save_success": False, "save_time_sec": 0.0, "file_size_kb": 0.0}
    start_time = time.time()
    global SELECTED_TOKEN_ATTRS # Access the global list decided in main
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dest = dest_path.with_suffix(dest_path.suffix + ".tmp")
        if temp_dest.exists(): os.remove(temp_dest)
        # <<< Use the globally selected attribute list >>>
        doc_bin = DocBin(attrs=SELECTED_TOKEN_ATTRS, docs=[doc], store_user_data=True)
        doc_bin.to_disk(str(temp_dest))
        temp_dest.rename(dest_path)
        metrics["save_success"] = True
        metrics["file_size_kb"] = dest_path.stat().st_size / 1024
        if config.force_sync: force_sync_directory(dest_path.parent)
    except Exception as e:
        error_msg = f"Failed to save DocBin to {dest_path}: {e}"
        logging.error(error_msg); metrics["error"] = error_msg
        if temp_dest.exists():
            try: os.remove(temp_dest)
            except OSError: pass
        if not config.disable_wandb and wandb.run:
            try:
                # Log save errors without step, as they aren't part of the main sequence
                wandb.log({"errors/document_save": wandb.Table(data=[[str(dest_path), str(e)]], columns=["Path", "Error Message"])})
            except Exception as log_e: logging.warning(f"Failed to log save error to W&B: {log_e}")
    finally:
        metrics["save_time_sec"] = time.time() - start_time
    return metrics

# (split_text_on_full_stop unchanged)
def split_text_on_full_stop(text: str, max_length: int) -> list:
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

# (log_system_metrics unchanged, already includes step)
def log_system_metrics(config: RunConfig, step: int):
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
        try: disk = psutil.disk_usage(str(config.dest_dir)); log_payload["sys/disk_util_percent"] = disk.percent; log_payload["sys/disk_free_gb"] = disk.free / (1024 ** 3)
        except FileNotFoundError: pass
        try: load_avg = os.getloadavg(); log_payload["sys/load_avg_1min"] = load_avg[0]
        except AttributeError: pass
        wandb.log(log_payload, step=step)
    except Exception as e: logging.warning(f"Could not log system metrics: {e}")

# (log_nlp_statistics unchanged, already includes step)
def log_nlp_statistics(doc: Doc, step: int, doc_id: Optional[str], config: RunConfig):
    if config.disable_wandb or not wandb.run: return
    stats_log = {}
    try:
        token_count = len(doc); stats_log["nlp/token_count"] = token_count
        if token_count == 0: return
        stats_log["nlp/unique_tokens"] = len(set([t.lower_ for t in doc]))
        stats_log["nlp/lexical_diversity"] = stats_log["nlp/unique_tokens"] / token_count if token_count > 0 else 0
        pos_counts = {}; ent_counts = {}
        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
            if token.ent_iob_ != "O": ent_counts[token.ent_type_] = ent_counts.get(token.ent_type_, 0) + 1
        if pos_counts:
             pos_data = [[pos, count, f"{(count/token_count*100):.1f}%"] for pos, count in sorted(pos_counts.items(), key=lambda i: i[1], reverse=True)]
             try: wandb.log({"nlp/pos_distribution_table": wandb.Table(data=pos_data, columns=["POS", "Count", "Percentage"])}, step=step)
             except Exception as tbl_e: logging.warning(f"Failed to log POS table: {tbl_e}")
        stats_log["nlp/entity_count"] = len(doc.ents)
        for ent_type, count in ent_counts.items(): stats_log[f"nlp/entities/{ent_type}"] = count
        try: sentences = list(doc.sents); sentence_count = len(sentences)
        except: sentences=[]; sentence_count=0
        stats_log["nlp/sentence_count"] = sentence_count
        stats_log["nlp/avg_sentence_len_tokens"] = token_count / sentence_count if sentence_count > 0 else 0
        wandb.log(stats_log, step=step)
    except Exception as e: logging.warning(f"Could not log NLP stats for {doc_id}: {e}")

# (manage_memory unchanged)
def manage_memory(step: int):
    try:
        start_mem = psutil.Process().memory_info().rss / (1024*1024); collected = gc.collect()
        end_mem = psutil.Process().memory_info().rss / (1024*1024)
        logging.info(f"[{step}] GC: Freed {collected} objs. Mem: {start_mem:.1f}MB -> {end_mem:.1f}MB")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception as e: logging.warning(f"Error during GC at step {step}: {e}")

# --- Core Processing Function ---
# (process_document unchanged - relies on save_document using global SELECTED_TOKEN_ATTRS)
def process_document(text: str, doc_id: str, dest_path: Path, nlp: Language, config: RunConfig) -> Tuple[bool, Dict[str, Any], Optional[Doc]]:
    logging.debug(f"[DEBUG {doc_id}] Enter process_document. Length: {len(text)}")
    start_time=time.time(); metrics = {"doc_id":doc_id,"doc_length":len(text),"had_to_split":False,"num_segments":1,"total_proc_time_sec":0.0,"split_time_sec":0.0,"pipe_time_sec":0.0,"combine_time_sec":0.0,"save_time_sec":0.0,"token_count":0,"error":None}; success=False; final_doc=None
    try:
        split_start = time.time(); logging.debug(f"[DEBUG {doc_id}] Check split (max={config.max_length})..."); texts=[text]
        if len(text) > config.max_length:
            logging.info(f"Doc '{doc_id}' len {len(text)} > max_len {config.max_length}. Split..."); metrics["had_to_split"]=True; texts=split_text_on_full_stop(text,config.max_length); metrics["num_segments"]=len(texts); logging.debug(f"[DEBUG {doc_id}] Split done. Segments: {metrics['num_segments']}")
            if not texts: logging.warning(f"Split '{doc_id}' got 0 segments."); texts=[]
        else: logging.debug(f"[DEBUG {doc_id}] No split needed.")
        metrics["split_time_sec"]=time.time()-split_start
        pipe_start=time.time(); docs: List[Doc]=[];
        if texts:
            non_empty=[t for t in texts if t]
            if non_empty:
                 n_proc=1 if len(non_empty)==1 else config.n_process; b_size=config.batch_size; logging.debug(f"[DEBUG {doc_id}] nlp.pipe: n={n_proc}, batch={b_size}, segs={len(non_empty)}...")
                 try: processed=nlp.pipe(non_empty,n_process=n_proc,batch_size=b_size); docs=list(processed); logging.debug(f"[DEBUG {doc_id}] nlp.pipe finished.")
                 except Exception as pipe_e: logging.error(f"nlp.pipe fail '{doc_id}': {pipe_e}"); metrics["error"]=f"pipe error: {pipe_e}"; docs=[]
            else: logging.warning(f"No non-empty segments for {doc_id}.")
        metrics["pipe_time_sec"]=time.time()-pipe_start
        logging.debug(f"[DEBUG {doc_id}] Combining {len(docs)} docs...")
        combine_start=time.time()
        if len(docs)>1:
            # Catch UserWarning W101 for trf_data when merging
            with warnings.catch_warnings():
                 warnings.filterwarnings("ignore", category=UserWarning, message=".*Skipping Doc custom extension 'trf_data'.*")
                 final_doc=Doc.from_docs(docs)
        elif len(docs)==1: final_doc=docs[0]
        else: final_doc=Doc(nlp.vocab); logging.warning(f"Created empty Doc for '{doc_id}'.")
        logging.debug(f"[DEBUG {doc_id}] Combine finished.")
        metrics["combine_time_sec"]=time.time()-combine_start
        metrics["token_count"]=len(final_doc) if final_doc else 0
        logging.debug(f"[DEBUG {doc_id}] Saving document...")
        save_metrics=save_document(final_doc, dest_path, config); metrics.update(save_metrics); logging.debug(f"[DEBUG {doc_id}] Save done. Success: {metrics['save_success']}")
        success=metrics["save_success"] and metrics["error"] is None
    except Exception as e:
        logging.exception(f"Critical error processing {doc_id}: {e}"); metrics["error"]=f"Critical: {e}"; success=False
        if not config.disable_wandb and wandb.run:
             try: wandb.log({"errors/critical_proc": wandb.Table(data=[[doc_id,str(dest_path),str(e)]],columns=["ID","Path","Error"])})
             except: pass # Ignore logging errors
    finally:
        metrics["total_proc_time_sec"]=time.time()-start_time; logging.debug(f"[DEBUG {doc_id}] Exit process_document. Time: {metrics['total_proc_time_sec']:.2f}s")
    return success, metrics, final_doc


# --- CLI Parser ---

def create_parser() -> argparse.ArgumentParser:
    """Creates parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="Corpus processor",
        description="Processes documents using spaCy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Paths
    parser.add_argument("--src_index", type=Path, required=True, help="Path to source CSV index.")
    parser.add_argument("--dest_dir", type=Path, required=True, help="Destination directory for .spacy files.")
    # Model
    parser.add_argument("--model", type=str, default="en_core_web_sm", dest="model_name", help="Name of spaCy model.")
    # Performance
    parser.add_argument("--max_length", type=int, default=RunConfig.max_length, help="Max doc length (chars) before splitting.")
    parser.add_argument("--n_process", type=int, default=RunConfig.n_process, help="Processes for nlp.pipe. TUNABLE.")
    parser.add_argument("--batch_size", type=int, default=RunConfig.batch_size, help="Batch size for nlp.pipe. TUNABLE.")
    parser.add_argument("--torch_threads", type=int, default=RunConfig.torch_threads, help="Max PyTorch threads.")
    # <<< ADDED ATTRIBUTES ARGUMENT >>>
    parser.add_argument("--attributes", type=str, default="full",
                        choices=["full", "ner_id", "ner_id_idx"],
                        dest="attributes_key", # Store choice key in config
                        help="Token attributes to save ('full', 'ner_id', 'ner_id_idx').")
    # Execution Control
    parser.add_argument("--force_sync", action="store_true", help="Force disk sync after saves (SLOW).")
    parser.add_argument("--sample_interval", type=int, default=RunConfig.sample_interval, help="Log detailed metrics every N docs.")
    parser.add_argument("--checkpoint_interval", type=int, default=RunConfig.checkpoint_interval, help="Save checkpoint every N docs.")
    # Test Mode
    parser.add_argument("--test_doc_id", type=str, default=None, help="Process only this document ID and exit.")
    parser.add_argument("--test_src_path", type=Path, default=None, help="Optional: Direct path to source text for test mode.")
    # W&B
    parser.add_argument("--wandb_user", type=str, default=os.getenv("WANDB_ENTITY", "sozialismus-au"), help="W&B user/entity.")
    parser.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT", "model-tracking"), help="W&B project.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional W&B run name.")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B logging.")

    return parser

# --- Main Execution ---

def main():
    """Main function to orchestrate corpus processing."""
    global SELECTED_TOKEN_ATTRS # Declare intent to modify global

    parser = create_parser()
    args = parser.parse_args()

    # --- Initialize Configuration ---
    try:
        # Initialize config directly from parsed args
        config = RunConfig(**vars(args))
    except Exception as config_e:
        logging.exception(f"Error initializing configuration: {config_e}")
        sys.exit(1)

    # --- Select Token Attributes based on Config ---
    # <<< SET GLOBAL BASED ON CONFIG >>>
    if config.attributes_key == "ner_id":
        SELECTED_TOKEN_ATTRS = TOKEN_ATTRS_NER_ID
    elif config.attributes_key == "ner_id_idx":
        SELECTED_TOKEN_ATTRS = TOKEN_ATTRS_NER_ID_IDX
    else: # Default or "full"
        SELECTED_TOKEN_ATTRS = TOKEN_ATTRS_FULL
    logging.info(f"Using token attribute set: '{config.attributes_key}' ({len(SELECTED_TOKEN_ATTRS)} attributes)")
    # ----------------------------------------------

    # Setup logging level for DEBUG prints if needed
    if config.test_doc_id:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Test mode activated, setting log level to DEBUG.")

    logging.info(f"Starting corpus processing run: {config.run_name}")
    logging.info(f"Using Config: {config}")

    # --- Directory Setup ---
    try:
        config.dest_dir.mkdir(exist_ok=True, parents=True)
        checkpoint_dir = config.dest_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        logging.info(f"Output directory: {config.dest_dir}")
        logging.info(f"Checkpoint directory: {checkpoint_dir}")
    except OSError as e: logging.exception(f"Could not create directories: {e}"); sys.exit(1)

    # --- Torch Thread Setup ---
    num_torch_threads_to_set = 1 # Apply deadlock workaround
    logging.info(f"Setting global PyTorch threads to: {num_torch_threads_to_set} (Workaround)")
    torch.set_num_threads(num_torch_threads_to_set)
    config.torch_threads = num_torch_threads_to_set # Update config to reflect actual value

    # --- Initialize W&B ---
    if not config.disable_wandb:
        if not config.wandb_user or not config.wandb_project:
             logging.warning("W&B user/project not set. Disabling W&B.")
             config.disable_wandb = True
        else:
            logging.info(f"Initializing W&B (User: {config.wandb_user}, Project: {config.wandb_project}, Run: {config.run_name})...")
            try:
                wandb.init(project=config.wandb_project, entity=config.wandb_user, name=config.run_name,
                           tags=["corpus-processing", config.model_name, "single-model", config.attributes_key]) # Add attr key tag
                config.log_to_wandb()
            except Exception as wandb_e:
                 logging.error(f"Error initializing WandB: {wandb_e}. Disabling W&B."); config.disable_wandb = True
    else: logging.info("W&B logging explicitly disabled.")

    # --- Load spaCy Model ---
    # (Model loading unchanged)
    logging.info(f"Loading spaCy model: {config.model_name}...")
    nlp: Optional[Language] = None; start_time_model_load = time.time()
    try:
        with warnings.catch_warnings():
             warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
             nlp = spacy.load(config.model_name, exclude=[])
        nlp.max_length = config.max_length
        model_load_time = time.time() - start_time_model_load; mem_after_load = psutil.Process().memory_info().rss / (1024*1024)
        logging.info(f"Model loaded in {model_load_time:.2f}s. Memory: {mem_after_load:.1f}MB")
        if not config.disable_wandb and wandb.run:
            wandb.log({"perf/model_load_time_sec": model_load_time, "mem/after_load_mb": mem_after_load}, step=0) # Log at step 0
            try: wandb.summary["model/pipeline_components"] = nlp.pipe_names
            except: pass
    except Exception as e:
        logging.exception(f"Failed to load model '{config.model_name}'. Exiting.")
        if not config.disable_wandb and wandb.run:
            try: wandb.log({"errors/critical": wandb.Table(data=[["Model Loading",str(e)]],columns=["Stage","Error"])})
            finally: wandb.finish(exit_code=1)
        sys.exit(1)


    # --- Load Index ---
    # (Index loading unchanged)
    logging.info(f"Loading index: {config.src_index}")
    parsed_index: Optional[pd.DataFrame] = None; n_total_in_index = 0
    try:
        if not config.src_index.is_file(): raise FileNotFoundError(f"Index not found: {config.src_index}")
        parsed_index = pd.read_csv(config.src_index)
        req_cols=['document_id','dest_path'];
        if not all(c in parsed_index.columns for c in req_cols): raise ValueError(f"Index needs columns: {req_cols}")
        parsed_index['document_id']=parsed_index['document_id'].astype(str)
        if parsed_index['document_id'].duplicated().any(): logging.warning("Duplicate IDs in index.")
        n_total_in_index=len(parsed_index.index); logging.info(f"Loaded index: {n_total_in_index} docs.")
        if not config.disable_wandb and wandb.run: wandb.log({"data/total_docs_in_index": n_total_in_index}, step=0) # Log at step 0
    except Exception as e:
        if config.test_doc_id and config.test_src_path: logging.warning(f"Failed load index, continue test mode: {e}"); parsed_index=None
        else: logging.exception(f"Failed load index '{config.src_index}'. Exit."); sys.exit(1)


    # ========================== TEST MODE BLOCK ==============================
    # (Test mode logic unchanged)
    if config.test_doc_id:
        logging.info("*** RUNNING SINGLE FILE TEST ***"); TEST_DOC_ID=config.test_doc_id; TEST_SRC_PATH=config.test_src_path
        if not TEST_SRC_PATH:
            if parsed_index is None: logging.error("Index failed & --test_src_path not given."); sys.exit(1)
            logging.info(f"Lookup path for ID '{TEST_DOC_ID}'..."); doc_row=parsed_index[parsed_index['document_id']==TEST_DOC_ID]
            if doc_row.empty: logging.error(f"Test ID '{TEST_DOC_ID}' not in index."); sys.exit(1)
            TEST_SRC_PATH=Path(doc_row.iloc[0]['dest_path']); logging.info(f"Found path: {TEST_SRC_PATH}")
        TEST_DEST_PATH=config.dest_dir/f"{TEST_DOC_ID}.spacy"; logging.info(f"Test Config: ID={TEST_DOC_ID}, Src={TEST_SRC_PATH}, Dest={TEST_DEST_PATH}, NProc={config.n_process}, Batch={config.batch_size}, MaxLen={config.max_length}, Attrs='{config.attributes_key}'")
        try:
            if not TEST_SRC_PATH.is_file(): logging.error(f"Test file not found: {TEST_SRC_PATH}"); sys.exit(1)
            with open(TEST_SRC_PATH,'r',encoding='utf-8') as f: test_text=f.read()
            logging.info(f"Read {len(test_text):,} chars.");
            if len(test_text)<=config.max_length: logging.warning("Test file not > max_length.")
        except Exception as e: logging.exception(f"Failed read test file {TEST_SRC_PATH}"); sys.exit(1)
        logging.info(f"Starting processing test file...")
        test_success,test_metrics,test_final_doc=process_document(test_text,TEST_DOC_ID,TEST_DEST_PATH,nlp,config)
        print("\n"+"***"*10+" TEST COMPLETE "+"***"*10); print(f"  Success: {test_success}"); print(f"  Metrics:")
        for k,v in test_metrics.items(): print(f"    {k:<30}: {v:.4f}" if isinstance(v,float) else f"    {k:<30}: {v}")
        if test_success and test_final_doc and not config.disable_wandb and wandb.run:
             logging.info("Logging NLP stats for test doc..."); log_nlp_statistics(test_final_doc,0,TEST_DOC_ID,config)
        logging.info("Exiting after test.");
        if not config.disable_wandb and wandb.run: wandb.summary["test_mode_results"]=test_metrics; wandb.summary["test_mode_success"]=test_success; wandb.finish(exit_code=0)
        sys.exit(0)
    # ======================= END OF TEST MODE BLOCK ==========================


    # --- Filter Processed Files (Full Run) ---
    # (Filtering unchanged)
    logging.info(f"Check processed in {config.dest_dir}..."); done_ids=get_done_ids(config.dest_dir); n_done=len(done_ids)
    parsed_index_filtered = parsed_index
    if n_done > 0:
        done_filter=parsed_index['document_id'].isin(done_ids); n_already_done=done_filter.sum()
        logging.info(f"Ignoring {n_already_done} indexed docs with existing output.")
        parsed_index_filtered=parsed_index[~done_filter].copy()
    else: logging.info("No previously processed files found.")
    n_left=len(parsed_index_filtered.index)
    if n_left == 0: logging.info("No docs left to process."); sys.exit(0)
    logging.info(f"Docs to process: {n_left}")
    if not config.disable_wandb and wandb.run: wandb.log({"data/docs_to_process":n_left,"data/docs_already_done":n_done,"progress/initial_pie":Plotly(progress_piechart(n_done,n_left+n_done))}, step=0)


    # --- Prepare Iteration Data (Full Run) ---
    # (Prep unchanged)
    logging.info("Preparing data for iteration...");
    try:
        src_paths=parsed_index_filtered['dest_path'].tolist(); doc_ids=parsed_index_filtered['document_id'].tolist(); doc_filenames=[config.dest_dir/f"{did}.spacy" for did in doc_ids]; texts_stream=stream_files(src_paths); logging.info("Data prepared.")
    except KeyError as e: logging.exception(f"Missing column in index: {e}"); sys.exit(1)
    except Exception as e: logging.exception(f"Error preparing iter data: {e}"); sys.exit(1)


    # --- Initialize Tracking (Full Run) ---
    start_time_total=time.time(); processed_chars_total=0; successful_docs=0; failed_docs=0
    total_processed_in_loop=0; all_metrics: List[Dict]=[]; last_successful_doc: Optional[Doc]=None

    # --- Main Processing Loop (Full Run) ---
    logging.info(f"Starting processing {n_left} documents...")
    pbar=tqdm(zip(doc_filenames,texts_stream,doc_ids),total=n_left,desc="Processing Docs",unit="doc")
    try:
        for doc_out_path,text,doc_id in pbar:
            loop_start=time.time(); current_step=n_done+total_processed_in_loop+1
            logging.info(f"Processing Doc ID: {doc_id} ({total_processed_in_loop+1}/{n_left})... Step: {current_step}")
            success,metrics,final_doc=process_document(text,doc_id,doc_out_path,nlp,config); all_metrics.append(metrics)
            if success and final_doc: last_successful_doc=final_doc
            total_processed_in_loop+=1; doc_len=metrics.get("doc_length",0);
            if doc_len>0: processed_chars_total+=doc_len
            if success: successful_docs+=1
            else: failed_docs+=1
            loop_time=time.time()-loop_start
            if not config.disable_wandb and wandb.run:
                elapsed=time.time()-start_time_total; docs_ps=total_processed_in_loop/elapsed if elapsed>0 else 0.0; chars_ps=processed_chars_total/elapsed if elapsed>0 else 0.0; est_rem_s=(elapsed/total_processed_in_loop)*(n_left-total_processed_in_loop) if total_processed_in_loop>0 else 0.0
                log_data={"progress/n_total":current_step,"progress/n_this_run":total_processed_in_loop,"progress/percent":(current_step/(n_left+n_done))*100 if (n_left+n_done)>0 else 0.0,"progress/id":doc_id,"perf/len":metrics.get("doc_length",0),"perf/proc_sec":metrics.get("total_proc_time_sec",0.0),"perf/save_sec":metrics.get("save_time_sec",0.0),"perf/pipe_sec":metrics.get("pipe_time_sec",0.0),"perf/docs_ps":docs_ps,"perf/chars_ps":chars_ps,"time/elapsed_min":elapsed/60,"time/est_rem_min":est_rem_s/60,"stats/ok":successful_docs,"stats/fail":failed_docs,"stats/tokens":metrics.get("token_count",0),"stats/fsize_kb":metrics.get("file_size_kb",0.0),"perf/loop_sec":loop_time,}
                if total_processed_in_loop%20==0 or total_processed_in_loop==n_left:
                    try: log_data["progress/pie"]=Plotly(progress_piechart(current_step,n_left+n_done))
                    except Exception as plot_e: logging.warning(f"Pie chart failed: {plot_e}")
                try: wandb.log(log_data,step=current_step)
                except Exception as log_e: logging.warning(f"W&B log fail step {current_step}: {log_e}")
            if total_processed_in_loop%config.checkpoint_interval==0 and total_processed_in_loop>0:
                logging.info(f"[{total_processed_in_loop}] Saving checkpoint..."); cp_path=checkpoint_dir/f"checkpoint_{current_step}.csv"
                try:
                    pd.DataFrame(all_metrics).assign(timestamp=time.time()).to_csv(cp_path,index=False); logging.info(f"Checkpoint -> {cp_path}")
                    if not config.disable_wandb and wandb.run:
                        try: art=wandb.Artifact(f"checkpoint_{current_step}",type="checkpoint"); art.add_file(str(cp_path)); wandb.log_artifact(art)
                        except Exception as log_e: logging.warning(f"Log artifact fail: {log_e}")
                except Exception as cp_e: logging.error(f"Checkpoint save fail: {cp_e}")
            if total_processed_in_loop%config.sample_interval==0 and total_processed_in_loop>0:
                logging.info(f"[{total_processed_in_loop}] Logging detailed metrics..."); log_system_metrics(config,step=current_step)
                if last_successful_doc: log_nlp_statistics(last_successful_doc,current_step,last_successful_doc._.get('doc_id','N/A'),config)
                else: logging.info(f"[{total_processed_in_loop}] Skip NLP stats log.")
            if total_processed_in_loop%(config.checkpoint_interval//2+1)==0 and total_processed_in_loop>0: manage_memory(total_processed_in_loop)
            pbar.set_description(f"Last: {doc_id[:15]} ({metrics.get('total_proc_time_sec',0.0):.2f}s)")
    except KeyboardInterrupt: logging.warning("\nInterrupted. Exiting loop.")
    finally: pbar.close(); logging.info("Loop finished/interrupted.")

    # --- Final Index and Checkpoint ---
    # (Final save logic unchanged)
    logging.info("Saving final index & metrics...");
    if not all_metrics: logging.warning("No docs processed, skip final save.")
    else:
        try:
            final_df=pd.DataFrame(all_metrics); final_df['status']=final_df['error'].apply(lambda x: 'failed' if pd.notna(x) else 'success')
            if parsed_index is not None: id_map=pd.Series(parsed_index['dest_path'].values,index=parsed_index['document_id']).to_dict(); final_df['src_path']=final_df['doc_id'].map(id_map)
            else: final_df['src_path']='N/A'
            final_df['dest_path']=final_df['doc_id'].map(lambda did: str(config.dest_dir/f"{did}.spacy"))
            idx_cols=['doc_id','src_path','dest_path','status','doc_length','token_count','total_proc_time_sec']; final_idx=final_df.reindex(columns=idx_cols,fill_value=None)
            idx_path=config.dest_dir/f"index_{config._run_id}.csv"; final_idx.to_csv(idx_path,index=False); logging.info(f"Saved index ({len(final_idx)}) -> {idx_path}")
            if not config.disable_wandb and wandb.run:
                 try: art=wandb.Artifact(f"index_{config.run_name}",type="dataset_index"); art.add_file(str(idx_path)); wandb.log_artifact(art)
                 except Exception as log_e: logging.warning(f"Log index artifact fail: {log_e}")
            final_cp_path=checkpoint_dir/f"final_metrics_{config._run_id}.csv"; final_df.to_csv(final_cp_path,index=False); logging.info(f"Saved final metrics ({len(final_df)}) -> {final_cp_path}")
            if not config.disable_wandb and wandb.run:
                try: art=wandb.Artifact(f"metrics_{config.run_name}",type="run_metrics"); art.add_file(str(final_cp_path)); wandb.log_artifact(art)
                except Exception as log_e: logging.warning(f"Log metrics artifact fail: {log_e}")
        except Exception as final_e: logging.exception(f"Final save error: {final_e}")

    # --- Final Summary ---
    # (Summary logic unchanged)
    logging.info("Calculating final summary..."); t_total_s=time.time()-start_time_total; n_actual=total_processed_in_loop
    rate_float=(successful_docs/n_actual*100) if n_actual>0 else 0.0; avg_t_s=(t_total_s/n_actual) if n_actual>0 else 0.0; avg_ch_s=(processed_chars_total/t_total_s) if t_total_s>0 else 0.0
    summary=[("Index Total",n_total_in_index if parsed_index is not None else "N/A"),("Skipped (Done)",n_done),("Attempted",n_actual),("Successful",successful_docs),("Failed",failed_docs),("Success Rate",rate_float),("Total Time (min)",t_total_s/60),("Avg Time/Doc (s)",avg_t_s),("Avg Speed (char/s)",avg_ch_s),("Total Chars",processed_chars_total),("Conf: nlp.pipe Procs",config.n_process),("Conf: nlp.pipe Batch",config.batch_size),("Conf: Max Length",config.max_length),("Conf: Torch Threads",config.torch_threads),("Conf: Attrs", config.attributes_key)] # Added Attrs key to summary
    print("\n"+"="*25+" Processing Summary "+"="*25)
    for name,val in summary: print(f"{name:<45}: {val:.2f}%" if name=="Success Rate" else (f"{val:.2f}" if isinstance(val,float) else f"{val}"))
    print("="*70)
    if not config.disable_wandb and wandb.run:
        logging.info("Logging final summary to W&B...")
        try:
            wandb.log({"final_summary": wandb.Table(data=[[i[0],i[1]] for i in summary],columns=["Metric","Value"])})
            wandb.log({"perf/avg_doc_time_sec":avg_t_s,"perf/avg_chars_per_sec":avg_ch_s,"stats/success_rate_percent":rate_float,"time/total_runtime_min":t_total_s/60,"stats/total_successful":successful_docs,"stats/total_failed":failed_docs})
            logging.info("Finishing W&B run..."); wandb.finish(); logging.info("W&B run finished.")
        except Exception as log_e: logging.error(f"Final W&B log/finish error: {log_e}")

    logging.info("Script finished.")


if __name__ == "__main__":
    main()
