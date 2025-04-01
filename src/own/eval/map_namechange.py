import pandas as pd
import argparse
import re
import logging
import json
from collections import Counter

# Set up logging to file and console.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("id_changes.log"),
        logging.StreamHandler()
    ]
)

def parse_doc_id(doc_id):
    """
    Parse a document ID into its components.
    
    Expected format examples:
      first1k_tlg0057.tlg073.1st1K-grc1
      first1k_tlg0090.tlg001.opp-grc10
      perseus_tlg0004.tlg001.perseus-grc1
      
    This function returns a dict with keys:
      - corpus: the corpus identifier (if present), e.g. "first1k", "perseus", etc.
      - tlg: the stable textual identifier, e.g. "tlg0057.tlg073"
      - category: the subcategory after the dot (e.g. "1st1K" or "opp"), may be empty\n      - version: the version indicator (e.g. "grc1", "grc2", etc.)
    """
    # Regex pattern breakdown:
    #   ^(?P<corpus>first1k|perseus|digicorpus)?_?    -> optional corpus prefix
    #   (?P<tlg>tlg\d+\.\w+)                           -> the tlg id (e.g., tlg0057.tlg073)
    #   (?:\.(?P<category>[^-]+))?                      -> optional category, until the hyphen
    #   -(?P<version>grc\d+)$                          -> version indicator
    pattern = r'^(?P<corpus>first1k|perseus|digicorpus)?_?(?P<tlg>tlg\d+\.\w+)(?:\.(?P<category>[^-]+))?-(?P<version>grc\d+)$'
    m = re.match(pattern, doc_id.lower())
    if m:
        return m.groupdict()
    else:
        logging.error(f"Failed to parse document ID: {doc_id}")
        return {"corpus": None, "tlg": None, "category": None, "version": None}

def compare_doc_ids(old_dict, new_dict):
    """
    Compare two parsed document ID dictionaries and return a description of the differences.
    """
    changes = []
    for key in ["corpus", "tlg", "category", "version"]:
        old_val = old_dict.get(key)
        new_val = new_dict.get(key)
        if old_val != new_val:
            changes.append(f"{key} changed from '{old_val}' to '{new_val}'")
    if changes:
        return "; ".join(changes)
    else:
        return "no change"

def map_id_changes(csv_path, output_csv):
    """
    Reads a CSV file containing columns 'old_document_id' and 'new_document_id',
    parses each ID into components, compares them, and writes out a CSV file that includes:
      - the original old_document_id and new_document_id
      - the parsed components (as JSON strings for clarity)
      - a change_description describing what changed.
      
    Also prints summary statistics on the types of changes.
    """
    # Load the CSV file.
    df = pd.read_csv(csv_path)
    if "old_document_id" not in df.columns or "new_document_id" not in df.columns:
        raise ValueError("CSV must contain columns 'old_document_id' and 'new_document_id'")
    
    # Parse both old and new IDs.
    df["old_parsed"] = df["old_document_id"].apply(parse_doc_id)
    df["new_parsed"] = df["new_document_id"].apply(parse_doc_id)
    
    # Compute a change description.
    df["change_description"] = df.apply(lambda row: compare_doc_ids(row["old_parsed"], row["new_parsed"]), axis=1)
    
    # For clarity, also store the parsed versions as JSON strings.
    df["old_parsed_json"] = df["old_parsed"].apply(lambda d: json.dumps(d, ensure_ascii=False))
    df["new_parsed_json"] = df["new_parsed"].apply(lambda d: json.dumps(d, ensure_ascii=False))
    
    # Select the columns to output.
    output_columns = ["old_document_id", "new_document_id", "old_parsed_json", "new_parsed_json", "change_description"]
    output_df = df[output_columns].copy()
    
    # Write the output CSV.
    output_df.to_csv(output_csv, index=False)
    print(f"🎉 Mapping complete! Changed document IDs written to {output_csv}")
    
    # Generate summary statistics on change types.
    change_counter = Counter(df["change_description"])
    print("\nSummary of changes:")
    for change, count in change_counter.items():
        print(f"{change}: {count}")

def main():
    parser = argparse.ArgumentParser(
        description="Map old document IDs to new ones by comparing naming components and output a detailed CSV."
    )
    parser.add_argument("csv_file", help="Path to the CSV file with old and new document IDs.")
    parser.add_argument("output_csv", help="Path for the output CSV file with mapping details.")
    args = parser.parse_args()
    
    map_id_changes(args.csv_file, args.output_csv)

if __name__ == "__main__":
    main()
