# regenerate_fullstop.py
import argparse
import logging
import sys
import os
import re
from pathlib import Path
import tqdm # For progress bar

# --- Configure standard logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
script_logger = logging.getLogger("FullstopRegenerator")

def create_parser() -> argparse.ArgumentParser:
    """Creates parser for the fullstop regeneration utility."""
    parser = argparse.ArgumentParser(
        description="Recursively finds *-joined.txt files and regenerates corresponding *-fullstop.txt files, adding newlines ONLY after periods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--base-dir", type=Path, required=True,
        help="Path to the base directory containing the reorganized corpus output (e.g., the main output directory)."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing *-fullstop.txt files. If False, only generates missing files."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Perform all steps except writing the output files. Logs what would be done."
    )
    return parser

def generate_fullstop_text(joined_text: str) -> str:
    """
    Applies logic to convert joined text into fullstop format, adding
    newlines ONLY after periods followed by whitespace.
    """
    if not joined_text:
        return ""

    # --- MODIFIED Logic ---
    # Only match a literal period (.) followed by whitespace (\s+)
    # The period needs to be escaped with a backslash (\.) because '.' is a special regex character.
    # Replace the whitespace with a newline (\n), keeping the period (\1 refers to the captured group).
    tfs = re.sub(r'(\.)\s+', r'\1\n', joined_text)
    # --- End Modification ---


    # Cleanup: Remove any remaining leading/trailing whitespace on each line
    # and ensure single newlines between sentences/segments.
    lines = tfs.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()] # Remove empty lines
    tfs = "\n".join(cleaned_lines)

    return tfs

def main():
    parser = create_parser()
    args = parser.parse_args()

    script_logger.info("--- Starting Fullstop File Regeneration (Periods Only) ---") # Updated title
    script_logger.info(f"Base Directory: {args.base_dir.resolve()}")
    script_logger.info(f"Overwrite existing files: {args.overwrite}")
    script_logger.info(f"Dry Run: {args.dry_run}")

    if not args.base_dir.is_dir():
        script_logger.error(f"Base directory not found or not a directory: {args.base_dir}")
        sys.exit(1)

    # Find all *-joined.txt files recursively
    script_logger.info("Scanning for *-joined.txt files...")
    joined_files = list(args.base_dir.rglob("*-joined.txt"))
    script_logger.info(f"Found {len(joined_files)} *-joined.txt files.")

    if not joined_files:
        script_logger.info("No files found to process. Exiting.")
        sys.exit(0)

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for joined_path in tqdm.tqdm(joined_files, desc="Regenerating Fullstop Files", unit="file"):
        try:
            # Derive the corresponding fullstop file path
            fullstop_filename = joined_path.name.replace("-joined.txt", "-fullstop.txt")
            fullstop_path = joined_path.parent / fullstop_filename

            if fullstop_path.exists() and not args.overwrite and not args.dry_run:
                script_logger.debug(f"Skipping (exists, no overwrite): {fullstop_path}")
                skipped_count += 1
                continue

            # Read the joined text
            try:
                with open(joined_path, 'r', encoding='utf-8') as f_in:
                    joined_content = f_in.read()
            except FileNotFoundError:
                script_logger.warning(f"Joined file disappeared before reading: {joined_path}")
                error_count += 1
                continue
            except Exception as read_err:
                script_logger.error(f"Error reading {joined_path}: {read_err}")
                error_count += 1
                continue

            # Generate the fullstop text using the corrected logic (periods only)
            fullstop_content = generate_fullstop_text(joined_content)

            # Write the fullstop text
            if args.dry_run:
                script_logger.info(f"[Dry Run] Would write {len(fullstop_content)} chars to {fullstop_path}")
            else:
                try:
                    with open(fullstop_path, 'w', encoding='utf-8') as f_out:
                        f_out.write(fullstop_content)
                    script_logger.debug(f"Successfully generated: {fullstop_path}")
                except Exception as write_err:
                    script_logger.error(f"Error writing {fullstop_path}: {write_err}")
                    error_count += 1
                    continue # Skip incrementing processed_count on write error

            processed_count += 1 # Count successful processing (or successful dry run simulation)

        except Exception as e:
            script_logger.error(f"Unexpected error processing {joined_path}: {e}", exc_info=True)
            error_count += 1

    script_logger.info("--- Regeneration Summary ---")
    script_logger.info(f"Total *-joined.txt files found: {len(joined_files)}")
    script_logger.info(f"Successfully processed/regenerated: {processed_count}")
    script_logger.info(f"Skipped (exists, no overwrite): {skipped_count}")
    script_logger.info(f"Errors encountered: {error_count}")
    if args.dry_run:
        script_logger.warning("NOTE: Dry run was active. No files were actually modified.")
    script_logger.info("--- Fullstop File Regeneration Complete ---")

if __name__ == "__main__":
    main()
