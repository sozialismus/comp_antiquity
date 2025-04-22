"""Script responsible for cleaning/processing the corpus."""
import argparse
import glob
import multiprocessing
import os
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
import psutil
import spacy
import torch
import wandb
from spacy.language import Language
from spacy.tokens import Doc, DocBin
from tqdm import tqdm
from utils.streams import stream_files
from wandb.data_types import Plotly


# Adaptive system resource configuration
def configure_system_resources():
    """Configure system resources based on available hardware"""
    cpu_count = os.cpu_count()
    available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
    
    # Implement the requested processor scaling logic:
    # - For systems with more than 8 cores: use 25% of cores
    # - For systems with 8 or fewer cores: use 75% of cores
    if cpu_count > 8:
        n_process = max(1, int(cpu_count * 0.25))
    else:
        n_process = max(1, int(cpu_count * 0.75))
    
    # Adaptive batch size and max length based on available memory
    batch_size = max(1, min(int(available_memory_gb * 2), 64))  # Scale batch size with memory
    max_length = min(10**4, int(available_memory_gb * 5 * 10**3))  # Scale max length with memory
    
    return {
        "MAX_LENGTH": max_length,
        "N_PROCESS": n_process,
        "BATCH_SIZE": batch_size,
        "CPU_COUNT": cpu_count,
        "AVAILABLE_MEMORY_GB": available_memory_gb,
        "TORCH_THREADS": max(1, cpu_count // 4)  # More conservative than original
    }


# Get system-specific parameters
SYSTEM_CONFIG = configure_system_resources()
MAX_LENGTH = SYSTEM_CONFIG["MAX_LENGTH"]
N_PROCESS = SYSTEM_CONFIG["N_PROCESS"]
BATCH_SIZE = SYSTEM_CONFIG["BATCH_SIZE"]

TOKEN_ATTRS = [
    "IS_ALPHA",
    "IS_ASCII",
    "IS_DIGIT",
    "IS_LOWER",
    "IS_PUNCT",
    "IS_SPACE",
    "IS_TITLE",
    "IS_UPPER",
    "LIKE_URL",
    "LIKE_NUM",
    "LIKE_EMAIL",
    "IS_STOP",
    "IS_QUOTE",
    "IS_BRACKET",
    "IS_LEFT_PUNCT",
    "IS_RIGHT_PUNCT",
    "IS_CURRENCY",
    "ID",
    "ORTH",
    "LOWER",
    "NORM",
    "SHAPE",
    "PREFIX",
    "SUFFIX",
    "LENGTH",
    "LEMMA",
    "POS",
    "TAG",
    "DEP",
    "ENT_IOB",
    "ENT_TYPE",
    "ENT_ID",
    "ENT_KB_ID",
    "HEAD",
    "SENT_START",
    "SPACY",
    "LANG",
    "MORPH",
    "IDX",
]


def get_done_ids(path: str) -> List[str]:
    """Finds documents that have already been cleaned"""
    paths = glob.glob(os.path.join(path, "*"))
    filenames = [path.split("/")[-1] for path in paths]
    ids = [
        filename.split(".")[0]
        for filename in filenames
        if filename.endswith(".spacy")
    ]
    return ids


def progress_piechart(n_processed: int, n_total: int) -> go.Figure:
    """Draws piechart of progress"""
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["done", "left"],
                values=[n_processed, n_total - n_processed],
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
    """Force system to sync directory to disk"""
    try:
        if hasattr(os, 'sync'):  # Unix/Linux
            os.sync()
        elif os.name == 'nt':  # Windows
            import ctypes
            ctypes.windll.kernel32.FlushFileBuffers(ctypes.c_void_p(-1))
        
        # Additional platform-specific sync for directory
        if hasattr(os, 'fsync') and os.path.exists(directory_path):
            try:
                fd = os.open(directory_path, os.O_RDONLY)
                os.fsync(fd)
                os.close(fd)
            except Exception:
                pass  # Fallback silently if this specific approach fails
    except Exception as e:
        print(f"Warning: Could not force sync to disk: {e}")


def save_document(doc: Doc, dest: str) -> None:
    """Serializes and saves spaCy Document."""
    start_time = time.time()
    success = True
    error_msg = ""
    
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(dest)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Create and save DocBin
        doc_bin = DocBin(attrs=TOKEN_ATTRS, docs=[doc])
        doc_bin.to_disk(dest)
        
        # Force synchronization to ensure immediate write to disk
        force_sync_directory(directory)
        
    except Exception as e:
        success = False
        error_msg = str(e)
        wandb.log({"document_save_errors": wandb.Table(
            data=[[dest, str(e)]],
            columns=["Path", "Error Message"]
        )})
        raise
    finally:
        # Log save operation metrics
        save_time = time.time() - start_time
        wandb.log({
            "document_save_time": save_time,
            "document_save_success": success,
            "document_path": dest
        })


def split_text_on_full_stop(text: str, max_length: int) -> list:
    """
    Splits the text into chunks of at most max_length characters,
    preferring to split at full stops. If no full stop is found,
    it falls back to splitting at a newline, and if still not found,
    it splits exactly at max_length.
    """
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

    return segments


def log_nlp_statistics(doc, step, doc_id=None):
    """Log NLP statistics for a document to wandb"""
    try:
        # Token statistics
        token_count = len(doc)
        if token_count == 0:
            wandb.log({"nlp_stats_warnings": f"Document {doc_id} has 0 tokens"}, step=step)
            return
            
        unique_tokens = len(set([token.text.lower() for token in doc]))
        
        # Parts of speech distribution
        pos_counts = {}
        for token in doc:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        
        # Named entity recognition statistics
        ent_counts = {}
        for ent in doc.ents:
            ent_type = ent.label_
            ent_counts[ent_type] = ent_counts.get(ent_type, 0) + 1
        
        # Sentence statistics
        sentences = list(doc.sents)
        sentence_count = len(sentences)
        avg_sentence_length = token_count / sentence_count if sentence_count > 0 else 0
        
        # Token length statistics
        token_lengths = [len(token.text) for token in doc]
        avg_token_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        
        # Calculate sentiment if available in pipeline
        has_sentiment = "sentiment" in doc.user_data
        sentiment = doc.user_data.get("sentiment", 0) if has_sentiment else None
        
        # Create histogram data for sentence lengths
        if sentence_count > 0:
            sentence_lengths = [len(sent) for sent in sentences]
            sentence_length_bins = list(range(0, max(sentence_lengths) + 10, 5))
            sentence_length_hist = wandb.Histogram(
                sentence_lengths, 
                num_bins=min(20, len(sentence_length_bins))
            )
        else:
            sentence_length_hist = None
        
        # Log all statistics
        stats_log = {
            "token_count": token_count,
            "unique_tokens": unique_tokens,
            "lexical_diversity": unique_tokens / token_count if token_count > 0 else 0,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "avg_token_length": avg_token_length,
            "pos_distribution": wandb.Table(
                data=[[pos, count, count/token_count] for pos, count in pos_counts.items()],
                columns=["POS", "Count", "Percentage"]
            ),
            "entity_count": len(doc.ents),
        }
        
        # Only include sentiment if available
        if has_sentiment:
            stats_log["sentiment"] = sentiment
            
        # Only include sentence length histogram if sentences exist
        if sentence_length_hist:
            stats_log["sentence_length_histogram"] = sentence_length_hist
            
        # Only include entity distribution if entities exist
        if ent_counts:
            stats_log["entity_distribution"] = wandb.Table(
                data=[[ent_type, count, count/len(doc.ents) if len(doc.ents) > 0 else 0] 
                      for ent_type, count in ent_counts.items()],
                columns=["Entity Type", "Count", "Percentage"]
            )
        
        wandb.log(stats_log, step=step)
        
    except Exception as e:
        wandb.log({"nlp_stats_errors": wandb.Table(
            data=[[str(doc_id), str(e)]],
            columns=["Document ID", "Error"]
        )}, step=step)


def log_system_metrics():
    """Log system resource usage metrics"""
    try:
        # Get CPU stats
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_times = psutil.cpu_times_percent(interval=0.1)
        
        # Get memory stats
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Get disk stats for the current directory
        disk = psutil.disk_usage('/')
        
        # Get network stats if available
        try:
            net_io = psutil.net_io_counters()
            net_stats = {
                "net_bytes_sent": net_io.bytes_sent,
                "net_bytes_recv": net_io.bytes_recv,
                "net_packets_sent": net_io.packets_sent,
                "net_packets_recv": net_io.packets_recv,
            }
        except Exception:
            net_stats = {}
        
        # Log all metrics
        wandb.log({
            # CPU metrics
            "cpu_usage_percent": cpu_percent,
            "cpu_user_percent": cpu_times.user,
            "cpu_system_percent": cpu_times.system,
            "cpu_idle_percent": cpu_times.idle,
            
            # Memory metrics
            "memory_usage_percent": memory.percent,
            "memory_available_gb": memory.available / (1024 ** 3),
            "memory_used_gb": memory.used / (1024 ** 3),
            "memory_free_gb": memory.free / (1024 ** 3),
            "swap_usage_percent": swap.percent if hasattr(swap, 'percent') else 0,
            
            # Disk metrics
            "disk_usage_percent": disk.percent,
            "disk_free_gb": disk.free / (1024 ** 3),
            
            # Process metrics for this Python process
            "process_memory_mb": psutil.Process().memory_info().rss / (1024 * 1024),
            "process_cpu_percent": psutil.Process().cpu_percent(interval=0.1),
            "process_threads": psutil.Process().num_threads(),
            
            # System load
            "system_load_1min": os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0,
            "system_load_5min": os.getloadavg()[1] if hasattr(os, 'getloadavg') else 0,
            "system_load_15min": os.getloadavg()[2] if hasattr(os, 'getloadavg') else 0,
            
            # Network stats if available
            **net_stats
        })
    except Exception as e:
        wandb.log({"system_metrics_error": str(e)})


def process_document(text: str, nlp: Language, dest: str, doc_id: str = None) -> Tuple[bool, str, float, Dict]:
    """Turns text into a spaCy document.
    If the text is too long it is broken into lines and processed that way.
    Processes texts in parallel using spaCy's nlp.pipe() with n_process and batch_size.
    
    Returns:
        Tuple containing (success status, error message if any, processing time, metrics dict)
    """
    metrics = {
        "doc_length": len(text),
        "had_to_split": len(text) > MAX_LENGTH,
    }
    
    try:
        start_time = time.time()
        
        # Set CPU threads for PyTorch to avoid overloading
        torch.set_num_threads(SYSTEM_CONFIG["TORCH_THREADS"])
        
        # Track peak memory usage before processing
        peak_memory_before = psutil.Process().memory_info().rss / (1024 * 1024)

        if len(text) > MAX_LENGTH:
            # If the text is too long, break it into segments
            texts = split_text_on_full_stop(text, MAX_LENGTH)
            metrics["num_segments"] = len(texts)
            metrics["avg_segment_length"] = sum(len(t) for t in texts) / len(texts)
            
            wandb.log({"text_splitting": wandb.Table(
                data=[[doc_id, len(text), len(texts), MAX_LENGTH]],
                columns=["Document ID", "Original Length", "Number of Segments", "Max Length"]
            )})
        else:
            texts = [text]
            metrics["num_segments"] = 1
            metrics["avg_segment_length"] = len(text)

        # Use nlp.pipe() to process texts in parallel
        pipe_start_time = time.time()
        docs = list(nlp.pipe(texts, n_process=N_PROCESS, batch_size=BATCH_SIZE))
        pipe_time = time.time() - pipe_start_time
        metrics["pipe_processing_time"] = pipe_time
        
        # Track memory after processing
        peak_memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
        metrics["memory_usage_mb"] = peak_memory_after - peak_memory_before

        # Combine the processed segments back into a single document
        combine_start_time = time.time()
        doc = Doc.from_docs(docs)
        combine_time = time.time() - combine_start_time
        metrics["doc_combine_time"] = combine_time
        
        # Collect document statistics
        metrics["token_count"] = len(doc)
        metrics["entity_count"] = len(doc.ents)
        try:
            metrics["sentence_count"] = len(list(doc.sents))
        except Exception:
            metrics["sentence_count"] = 0
            
        # Save document and ensure it's written to disk
        save_start_time = time.time()
        save_document(doc, dest=dest)
        save_time = time.time() - save_start_time
        metrics["doc_save_time"] = save_time
        
        processing_time = time.time() - start_time
        metrics["total_processing_time"] = processing_time
        
        return True, "", processing_time, metrics
    except Exception as e:
        error_message = str(e)
        wandb.log({"processing_errors": wandb.Table(
            data=[[doc_id, dest, error_message]],
            columns=["Document ID", "Document Path", "Error Message"]
        )})
        return False, error_message, 0.0, metrics


def create_parser() -> argparse.ArgumentParser:
    """Creates parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="Corpus processor",
        description="Processes all documents in a corpus on CPU",
    )
    parser.add_argument("--model", type=str, default="grc_proiel_trf")
    parser.add_argument("--dest", type=str, default="dat/greek/processed_data/")
    parser.add_argument(
        "--src_index", type=str, default="dat/greek/cleaned_parsed_data/index.csv"
    )
    parser.add_argument("--wandb_user", type=str, default="sozialismus-au")
    parser.add_argument(
        "--wandb_project", type=str, default="model-tracking"
    )
    parser.add_argument("--run_name", type=str, default=None,
                       help="Optional name for the W&B run")
    parser.add_argument("--sample_interval", type=int, default=10,
                       help="Log detailed statistics every N documents")
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                       help="Save checkpoint data every N documents")
    parser.add_argument("--disable_wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    print(
        "--------------------------\n"
        "------PROCESS CORPUS------\n"
        "--------------------------\n"
    )

    # Creating destination directory
    print(f"Creating destination directory ({args.dest})")
    Path(args.dest).mkdir(exist_ok=True, parents=True)

    # Generate a descriptive run name if none provided
    run_name = args.run_name or f"corpus-processing-{args.model}-{time.strftime('%Y%m%d-%H%M%S')}"

    # Initialize wandb if not disabled
    if not args.disable_wandb:
        print(f"Initializing wandb with run name: {run_name}")
        wandb.init(
            project=args.wandb_project, 
            entity=args.wandb_user,
            name=run_name,
            tags=["corpus-processing", args.model]
        )

        # Log system configuration
        wandb.config.update({
            "model": args.model,
            "max_length": MAX_LENGTH,
            "n_process": N_PROCESS,
            "batch_size": BATCH_SIZE,
            "sample_interval": args.sample_interval,
            "checkpoint_interval": args.checkpoint_interval,
            "system_info": {
                "cpu_count": SYSTEM_CONFIG["CPU_COUNT"],
                "available_memory_gb": SYSTEM_CONFIG["AVAILABLE_MEMORY_GB"],
                "torch_threads": SYSTEM_CONFIG["TORCH_THREADS"],
                "device": "cpu",  # Update if using GPU
            }
        })
    else:
        print("Weights & Biases logging disabled")

    # Loading model
    print(f"Loading NLP model: {args.model}")
    start_time_model_load = time.time()
    try:
        nlp = spacy.load(args.model)
        # Set model max length to our calculated value
        nlp.max_length = MAX_LENGTH
        model_load_time = time.time() - start_time_model_load
        print(f"Model loaded in {model_load_time:.2f} seconds")
        
        if not args.disable_wandb:
            wandb.log({"model_load_time_seconds": model_load_time})
            # Log model pipeline components
            pipeline_components = [{"name": name, "type": str(type(component))} 
                                   for name, component in nlp.pipeline]
            wandb.log({"model_pipeline": wandb.Table(
                data=[[comp["name"], comp["type"]] for comp in pipeline_components],
                columns=["Component", "Type"]
            )})
    except Exception as e:
        error_message = f"Failed to load model: {str(e)}"
        print(error_message)
        if not args.disable_wandb:
            wandb.log({"critical_errors": wandb.Table(
                data=[["Model Loading", error_message]],
                columns=["Stage", "Error"]
            )})
        return

    # Loading Index
    print("Loading index of parsed files")
    try:
        parsed_index = pd.read_csv(args.src_index, index_col=0)
        n_total = len(parsed_index.index)
        print(f"Loaded index with {n_total} documents")
        
        if not args.disable_wandb:
            wandb.log({"total_documents": n_total})
    except Exception as e:
        error_message = f"Failed to load index: {str(e)}"
        print(error_message)
        if not args.disable_wandb:
            wandb.log({"critical_errors": wandb.Table(
                data=[["Index Loading", error_message]],
                columns=["Stage", "Error"]
            )})
        return

    # Removing texts from the index that have already been cleaned
    done_ids = get_done_ids(args.dest)
    done = parsed_index.document_id.isin(done_ids)
    n_done = done.sum()
    print(f"Ignoring previously completed documents (N={n_done})")
    parsed_index = parsed_index[~done]

    # Processing
    print("Processing texts")
    src_path = parsed_index.dest_path
    n_left = len(src_path)
    
    if not args.disable_wandb:
        wandb.log({
            "documents_to_process": n_left,
            "documents_already_processed": n_done,
            "initial_progress": Plotly(progress_piechart(n_done, n_total))
        })

    # Setting up file stream
    texts = stream_files(src_path)

    # Getting document ids from index
    doc_ids = parsed_index.document_id
    # Producing output file names
    doc_filenames = doc_ids.map(
        lambda doc_id: os.path.join(args.dest, f"{doc_id}.spacy")
    )

    # Initialize tracking variables
    start_time_total = time.time()
    processed_chars_total = 0
    successful_docs = 0
    failed_docs = 0
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.dest, "checkpoints")
    Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
    
    # Checkpoint dataframe for tracking processed documents
    checkpoint_df = pd.DataFrame(columns=["document_id", "status", "processing_time", "timestamp"])

    # Saving SpaCy documents
    for doc_out_path, text, doc_id, n_processed in zip(
        tqdm(doc_filenames), texts, doc_ids, range(n_left)
    ):
        doc_length = len(text)
        processed_chars_total += doc_length
        
        # Process the document
        current_doc_id = doc_id
        print(f"Processing document {n_processed+1}/{n_left}: {current_doc_id}")
        success, error_message, processing_time, metrics = process_document(
            text, nlp=nlp, dest=doc_out_path, doc_id=current_doc_id
        )
        
        # Update counters based on success
        if success:
            successful_docs += 1
            status = "success"
        else:
            failed_docs += 1
            status = "failed"
            if not args.disable_wandb:
                wandb.log({"failed_documents": wandb.Table(
                    data=[[current_doc_id, error_message]],
                    columns=["Document ID", "Error"]
                )})
        
        # Add to checkpoint dataframe
        checkpoint_df = checkpoint_df.append({
            "document_id": current_doc_id,
            "status": status,
            "processing_time": processing_time,
            "timestamp": time.time()
        }, ignore_index=True)
        
        # Save checkpoint at specified intervals
        if n_processed % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{n_processed}.csv")
            checkpoint_df.to_csv(checkpoint_path)
            print(f"Saved checkpoint at {checkpoint_path}")
            
            if not args.disable_wandb:
                checkpoint_artifact = wandb.Artifact(f"checkpoint_{n_processed}", type="dataset")
                checkpoint_artifact.add_file(checkpoint_path)
                wandb.log_artifact(checkpoint_artifact)
        
        # Calculate progress and timing metrics
        elapsed_time = time.time() - start_time_total
        docs_per_second = (n_processed + 1) / elapsed_time if elapsed_time > 0 else 0
        chars_per_second = processed_chars_total / elapsed_time if elapsed_time > 0 else 0
        
        if n_processed > 0:
            estimated_remaining_time = (elapsed_time / n_processed) * (n_left - n_processed)
        else:
            estimated_remaining_time = 0
        
        # Print progress to console
        print(f"Progress: {n_processed+1}/{n_left} documents, "
              f"Est. remaining: {estimated_remaining_time/60:.2f} minutes")
            
        # Log progress and document-specific metrics
        if not args.disable_wandb:
            log_data = {
                "n_processed": n_processed + n_done,
                "progress": Plotly(progress_piechart(n_processed + n_done, n_total)),
                "doc_length": doc_length,
                "processing_time_seconds": processing_time,
                "processing_speed_chars_per_sec": doc_length / processing_time if processing_time > 0 else 0,
                "successful_docs": successful_docs,
                "failed_docs": failed_docs,
                "elapsed_time_minutes": elapsed_time / 60,
                "docs_per_second": docs_per_second,
                "chars_per_second": chars_per_second,
                "estimated_remaining_time_minutes": estimated_remaining_time / 60,
                **metrics  # Include all metrics from process_document
            }
            wandb.log(log_data)
        
        # Log detailed statistics at specified intervals
        if not args.disable_wandb and n_processed % args.sample_interval == 0:
            # Log system metrics
            log_system_metrics()
            
            # Process a sample document for NLP statistics
            if success:
                try:
                    # Only analyze a sample of the text to avoid memory issues
                    sample_text = text[:min(len(text), MAX_LENGTH)]
                    sample_doc = nlp(sample_text)
                    log_nlp_statistics(sample_doc, n_processed, current_doc_id)
                except Exception as e:
                    wandb.log({"statistics_errors": wandb.Table(
                        data=[[current_doc_id, str(e)]],
                        columns=["Document ID", "Error"]
                    )})

    # Creating and saving index for cleaned documents
    index = pd.DataFrame(
        {
            "document_id": doc_ids,
            "dest_path": doc_filenames,
            "src_path": src_path,
            "processing_status": ["success" if doc_id in checkpoint_df[checkpoint_df["status"] == "success"]["document_id"].values else "failed" for doc_id in doc_ids]
        }
    )
    print("Saving index")
    index_path = os.path.join(args.dest, "index.csv")
    index.to_csv(index_path)
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint.csv")
    checkpoint_df.to_csv(final_checkpoint_path)
    
    if not args.disable_wandb:
        # Log index as an artifact
        index_artifact = wandb.Artifact(f"processed_corpus_index", type="dataset")
        index_artifact.add_file(index_path)
        wandb.log_artifact(index_artifact)
        
        # Log final checkpoint as an artifact
        final_checkpoint_artifact = wandb.Artifact("final_checkpoint", type="dataset")
        final_checkpoint_artifact.add_file(final_checkpoint_path)
        wandb.log_artifact(final_checkpoint_artifact)
    
    # After processing loop, build and print/log summary:
    processing_time_total = time.time() - start_time_total
    summary_data = [
        ["Total Documents", n_total],
        ["Previously Processed", n_done],
        ["Newly Processed", n_processed],
        ["Successfully Processed", successful_docs],
        ["Failed", failed_docs],
        ["Success Rate", f"{(successful_docs/n_processed*100):.2f}%" if n_processed > 0 else "N/A"],
        ["Total Processing Time (min)", f"{processing_time_total/60:.2f}"],
        ["Average Time per Document (sec)", f"{processing_time_total/n_processed:.2f}" if n_processed > 0 else "N/A"],
        ["Average Processing Speed (chars/sec)", f"{processed_chars_total/processing_time_total:.2f}" if processing_time_total > 0 else "N/A"],
        ["Total Characters Processed", processed_chars_total],
        ["System Configuration",
         f"CPUs: {SYSTEM_CONFIG['CPU_COUNT']}, Memory GB: {SYSTEM_CONFIG['AVAILABLE_MEMORY_GB']:.2f}, Batch Size: {BATCH_SIZE}, Processes: {N_PROCESS}"]
    ]

    # Print summary
    print("\n===== Processing Summary =====")
    for name, value in summary_data:
        print(f"{name}: {value}")

    # Log summary to W&B
    if not args.disable_wandb:
        wandb.log({"processing_summary": wandb.Table(
            data=summary_data,
            columns=["Metric", "Value"]
        )})
        wandb.finish()

if __name__ == "__main__":
    # Ensure start_time_total and other variables are in scope
    start_time_total = time.time()
    main()
