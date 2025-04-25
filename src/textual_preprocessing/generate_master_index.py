# generate_master_index.py
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import glob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def create_parser() -> argparse.ArgumentParser:
    """Creates parser for the index generation utility."""
    parser = argparse.ArgumentParser(
        description="Generates a master index of processing status for a spaCy corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--src_index", type=Path, required=True,
                        help="Path to the ORIGINAL source CSV index file listing ALL documents.")
    parser.add_argument("--processed_dir", type=Path, required=True,
                        help="Path to the destination directory WHERE .spacy files were saved.")
    parser.add_argument("--output_file", type=Path, required=True,
                        help="Path for the output master index CSV file.")
    parser.add_argument("--checkpoints_dir", type=Path, default=None,
                        help="Optional: Path to the checkpoints directory to read status from CSVs. "
                             "If not provided, status is based purely on .spacy file existence.")
    return parser

def get_processed_ids_from_spacy_files(processed_dir: Path) -> set:
    """Scans the directory for existing .spacy files and returns their IDs."""
    ids = set()
    if processed_dir.is_dir():
        # Use stem to get filename without suffix
        ids = {p.stem for p in processed_dir.glob("*.spacy") if p.is_file()}
    logging.info(f"Found {len(ids)} '.spacy' files in '{processed_dir}'.")
    return ids

def get_status_from_checkpoints(checkpoints_dir: Path) -> pd.DataFrame:
    """Loads status information from the latest checkpoint files."""
    all_checkpoint_data = []
    if checkpoints_dir and checkpoints_dir.is_dir():
        # Find all checkpoint files, sort by modification time (newest first)
        # Prioritize final_metrics files if they exist
        chkpt_files = sorted(
            list(checkpoints_dir.glob("final_metrics_*.csv")) + list(checkpoints_dir.glob("checkpoint_*.csv")),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if not chkpt_files:
            logging.warning(f"No checkpoint files found in {checkpoints_dir}.")
            return pd.DataFrame(columns=['document_id', 'processing_status'])

        logging.info(f"Found {len(chkpt_files)} checkpoint files. Loading relevant data...")

        processed_ids_in_checkpoints = set()
        # Load data, keeping only the latest status for each ID
        for cp_file in chkpt_files:
            try:
                # Read only necessary columns
                df = pd.read_csv(cp_file, usecols=['document_id', 'status', 'processing_status', 'error'],
                                 dtype={'document_id': str},
                                 on_bad_lines='warn') # Skip bad lines if any

                # Determine the correct status column (handle different checkpoint formats)
                if 'processing_status' in df.columns:
                    status_col = 'processing_status'
                elif 'status' in df.columns:
                    status_col = 'status'
                else: # Fallback: Infer from 'error' column if status cols missing
                     if 'error' in df.columns:
                         df['inferred_status'] = df['error'].apply(lambda x: 'failed' if pd.notna(x) else 'success')
                         status_col = 'inferred_status'
                     else:
                          logging.warning(f"Checkpoint {cp_file.name} missing status/processing_status/error columns. Cannot determine status.")
                          continue

                df.rename(columns={status_col: 'processing_status'}, inplace=True)

                # Filter out rows for IDs we already have a newer status for
                new_data = df[~df['document_id'].isin(processed_ids_in_checkpoints)][['document_id', 'processing_status']]
                all_checkpoint_data.append(new_data)
                processed_ids_in_checkpoints.update(new_data['document_id'])

            except Exception as e:
                logging.warning(f"Could not read or process checkpoint file {cp_file.name}: {e}")

        if all_checkpoint_data:
            status_df = pd.concat(all_checkpoint_data, ignore_index=True)
            # Ensure final uniqueness (in case multiple checkpoints had same latest mod time)
            status_df = status_df.drop_duplicates(subset=['document_id'], keep='first')
            logging.info(f"Loaded status for {len(status_df)} unique documents from checkpoints.")
            return status_df
        else:
            logging.warning("No valid data loaded from checkpoints.")
            return pd.DataFrame(columns=['document_id', 'processing_status'])

    else:
        logging.info("No checkpoints directory specified or found.")
        return pd.DataFrame(columns=['document_id', 'processing_status'])


def main():
    parser = create_parser()
    args = parser.parse_args()

    logging.info("--- Generating Master Processing Index ---")
    logging.info(f"Source Index: {args.src_index}")
    logging.info(f"Processed Dir: {args.processed_dir}")
    logging.info(f"Output File: {args.output_file}")
    if args.checkpoints_dir:
        logging.info(f"Checkpoints Dir: {args.checkpoints_dir}")
    else:
         logging.info("Checkpoints Dir: Not specified (status based on file existence).")

    # 1. Load Source Index
    try:
        if not args.src_index.is_file():
            raise FileNotFoundError(f"Source index file not found: {args.src_index}")
        master_df = pd.read_csv(args.src_index, dtype={'document_id': str})
        required_cols = ['document_id', 'dest_path']
        if not all(col in master_df.columns for col in required_cols):
             raise ValueError(f"Source index must contain columns: {required_cols}")
        master_df.rename(columns={'dest_path': 'src_path'}, inplace=True) # Rename for clarity
        logging.info(f"Loaded {len(master_df)} entries from source index.")
        # Ensure no duplicate document_ids in the source index itself
        if master_df['document_id'].duplicated().any():
             logging.warning("Source index contains duplicate document IDs! Keeping first occurrence.")
             master_df = master_df.drop_duplicates(subset=['document_id'], keep='first')

    except Exception as e:
        logging.exception(f"Failed to load source index {args.src_index}. Exiting.")
        sys.exit(1)

    # 2. Find existing .spacy files
    processed_ids = get_processed_ids_from_spacy_files(args.processed_dir)

    # 3. Get status from checkpoints (if specified)
    checkpoint_status_df = get_status_from_checkpoints(args.checkpoints_dir)

    # 4. Determine final status for each document
    status_list = []
    for doc_id in master_df['document_id']:
        status = "pending" # Default status
        # Check checkpoint status first (more reliable for failures)
        if not checkpoint_status_df.empty:
            match = checkpoint_status_df[checkpoint_status_df['document_id'] == doc_id]
            if not match.empty:
                # Ensure status is lowercase for consistency
                status = str(match.iloc[0]['processing_status']).lower()
                # If checkpoint says success, double-check file exists
                if status == 'success' and doc_id not in processed_ids:
                    logging.warning(f"Doc ID {doc_id} marked as 'success' in checkpoint but '.spacy' file not found.")
                    # Decide how to handle this: mark as failed? or keep success? Let's mark as potentially failed.
                    status = "failed (missing file)"
            elif doc_id in processed_ids:
                # Not found in checkpoint, but file exists - likely completed successfully in a run without checkpoints saved?
                status = "success (file exists)"
        elif doc_id in processed_ids:
             # No checkpoint info, rely solely on file existence
             status = "success (file exists)"

        status_list.append(status)

    master_df['processing_status'] = status_list
    # Add the expected destination path column
    master_df['processed_path'] = master_df['document_id'].apply(
        lambda did: str(args.processed_dir / f"{did}.spacy")
    )

    # Reorder columns for clarity
    output_cols = ['document_id', 'src_path', 'processed_path', 'processing_status']
    # Add any other columns from the original index you want to keep
    other_cols = [col for col in master_df.columns if col not in output_cols]
    master_df = master_df[output_cols + other_cols]

    # 5. Save the master index
    try:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        master_df.to_csv(args.output_file, index=False)
        logging.info(f"Master index saved successfully to: {args.output_file}")
        logging.info(f"Status Summary:\n{master_df['processing_status'].value_counts().to_string()}")
    except Exception as e:
        logging.exception(f"Failed to save master index to {args.output_file}.")
        sys.exit(1)

    logging.info("--- Master Index Generation Complete ---")


if __name__ == "__main__":
    main()
