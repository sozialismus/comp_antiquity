import pandas as pd
import argparse
import re
import logging

# Set up logging to file and console.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("docid_mapping.log"),
        logging.StreamHandler()
    ]
)

def parse_doc_id(doc_id):
    """
    Parses a document_id into its components:
      - corpus: the prefix (e.g., 'perseus' or 'first1k')
      - tlg: the stable TLG identifier (e.g., 'tlg0018.tlg031')
      - category: the part between the TLG identifier and the version (e.g., 'opp' or '1st1k'); may be empty
      - version: the trailing version indicator (e.g., 'grc1', 'grc2', etc.)
      
    Expected formats:
      first1k_tlg0018.tlg031.opp-grc1
      first1k_tlg0018.tlg031.1st1k-grc1
      perseus_tlg0004.tlg001.perseus-grc1
      
    Returns a dictionary with keys: corpus, tlg, category, version.
    """
    if not isinstance(doc_id, str) or not doc_id:
        return {"corpus": None, "tlg": None, "category": None, "version": None}
    
    # Regex explanation:
    #   ^(?P<corpus>[^_]+)_        -> everything up to the underscore (corpus)
    #   (?P<tlg>tlg\d+\.\w+)        -> the tlg id (e.g., tlg0018.tlg031)
    #   (?:\.(?P<category>[^-]+))?   -> optional category following a period (e.g., opp or 1st1k)
    #   -(?P<version>grc\d+)$       -> hyphen followed by the version indicator (e.g., grc1)
    pattern = r'^(?P<corpus>[^_]+)_(?P<tlg>tlg\d+\.\w+)(?:\.(?P<category>[^-]+))?-(?P<version>grc\d+)$'
    m = re.match(pattern, doc_id.lower())
    if m:
        d = m.groupdict()
        # If category is None, set it to empty string.
        if d["category"] is None:
            d["category"] = ""
        return d
    else:
        logging.error(f"Failed to parse document ID: {doc_id}")
        return {"corpus": None, "tlg": None, "category": None, "version": None}

def normalized_key(doc_id):
    """
    Returns a normalized key for comparison: the corpus plus the TLG id.
    This ignores the category and version.
    
    Example:
      "first1k_tlg0018.tlg031.opp-grc1"  -> "first1k_tlg0018.tlg031"
      "first1k_tlg0018.tlg031.1st1k-grc1" -> "first1k_tlg0018.tlg031"
    """
    parsed = parse_doc_id(doc_id)
    if parsed["corpus"] and parsed["tlg"]:
        return f"{parsed['corpus']}_{parsed['tlg']}"
    else:
        return doc_id.strip()

def compare_document_ids(master_csv, new_csv, output_csv):
    """
    Compares two CSV files (master and new) containing document IDs (and optionally metadata).
    It focuses only on document IDs that start with 'perseus_' or 'first1k_'.
    
    For each document in the master CSV, it determines if:
      - The text is unchanged (full document_id identical in both)
      - The text is updated (normalized key is the same but full document_id differs, e.g. category changed)
      - The text is removed (exists in master but not in new)
    
    It also flags texts that are new in the new CSV.
    
    Outputs a CSV that lists:
      - normalized_id
      - old_document_id (from master)
      - new_document_id (from new, if available)
      - old_category and new_category (for insight into what changed)
      - status: 'unchanged', 'updated', 'removed', or 'new'
      - (Optional: additional metadata from the master CSV)
    """
    # Read both CSV files.
    master_df = pd.read_csv(master_csv)
    new_df = pd.read_csv(new_csv)
    
    # Ensure document_id columns are strings.
    master_df["document_id"] = master_df["document_id"].fillna("").astype(str)
    new_df["document_id"] = new_df["document_id"].fillna("").astype(str)
    
    # Filter only rows with document_id starting with 'perseus_' or 'first1k_'
    master_df = master_df[master_df["document_id"].str.startswith(("perseus_", "first1k_"))].copy()
    new_df = new_df[new_df["document_id"].str.startswith(("perseus_", "first1k_"))].copy()
    
    # Create normalized keys.
    master_df["normalized_id"] = master_df["document_id"].apply(normalized_key)
    new_df["normalized_id"] = new_df["document_id"].apply(normalized_key)
    
    # Parse full IDs into components for later comparison.
    master_df["parsed"] = master_df["document_id"].apply(parse_doc_id)
    new_df["parsed"] = new_df["document_id"].apply(parse_doc_id)
    
    # Merge on normalized_id using an outer join.
    merged = pd.merge(master_df, new_df, on="normalized_id", how="outer",
                      suffixes=("_master", "_new"), indicator=True)
    
    def determine_status(row):
        if row["_merge"] == "left_only":
            return "removed"
        elif row["_merge"] == "right_only":
            return "new"
        elif row["_merge"] == "both":
            # If full document IDs differ, that's an update.
            if row["document_id_master"] != row["document_id_new"]:
                return "updated"
            else:
                return "unchanged"
        return "unknown"
    
    merged["status"] = merged.apply(determine_status, axis=1)
    
    # For further insight, extract old and new categories.
    def get_category(parsed):
        if isinstance(parsed, dict):
            return parsed.get("category", "")
        return ""
    
    merged["old_category"] = merged["parsed_master"].apply(get_category)
    merged["new_category"] = merged["parsed_new"].apply(get_category)
    
    # Prepare output: focus on texts from the master and their status.
    # We'll output rows where status is updated or removed from the master,
    # plus optionally, rows that are new in the new CSV.
    # Here, we'll output all rows.
    output_columns = [
        "normalized_id", "document_id_master", "document_id_new",
        "old_category", "new_category", "status"
    ]
    # Optionally, include metadata columns from master (if present).
    metadata_columns = ["english author", "english work title", "greek work title", "sort_id"]
    for col in metadata_columns:
        col_master = f"{col}_master"
        if col_master in merged.columns:
            output_columns.append(col_master)
    
    output_df = merged[output_columns].copy()
    
    # Write output CSV.
    output_df.to_csv(output_csv, index=False)
    
    # Log some summary counts.
    summary = merged["status"].value_counts().to_dict()
    logging.info(f"Summary of changes: {summary}")
    print(f"🎉 Extraction complete! Mapping written to {output_csv}")
    print("Summary of changes:", summary)

def main():
    parser = argparse.ArgumentParser(
        description="Compare two CSV files of document IDs to determine which texts have been updated or removed. "
                    "Focuses only on 'perseus_' and 'first1k_' texts."
    )
    parser.add_argument("master_csv", help="Path to the master CSV file (deprecated document IDs).")
    parser.add_argument("new_csv", help="Path to the new CSV file (current document IDs).")
    parser.add_argument("output_csv", help="Path for the output CSV file with mapping details.")
    args = parser.parse_args()
    
    compare_document_ids(args.master_csv, args.new_csv, args.output_csv)

# Alias function for clarity.
compare_document_ids = compare_document_ids  # Using the function above.

if __name__ == "__main__":
    main()
