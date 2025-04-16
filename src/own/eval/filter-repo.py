import os
import re

def remove_prefix_from_ids(doc_ids, prefix="first1k_"):
    """
    Removes the specified prefix from each document ID in the list.
    
    Parameters:
    - doc_ids: List of document IDs (strings).
    - prefix: The prefix string to remove from each document ID.
    
    Returns:
    - A new list of document IDs with the prefix removed.
    """
    return [doc_id[len(prefix):] if doc_id.startswith(prefix) else doc_id for doc_id in doc_ids]

# List of document_ids to check
document_ids = [
    "first1k_tlg0732.tlg006.opp-grc1",
    "first1k_tlg0732.tlg004.opp-grc1",
    "first1k_tlg0732.tlgX01.opp-grc1",
    "first1k_tlg1799.tlg007.First1K-grc1",
    "first1k_tlg3135.tlg004.opp-grc1",
    "first1k_tlg4021.tlg002.opp-grc1",
    "first1k_tlg0544.tlg001.opp-grc1",
    "first1k_tlg3135.tlg005.opp-grc1",
    "first1k_tlg2042.tlg005.opp-grc1",
    "first1k_tlg4084.tlg001.opp-grc1",
    "first1k_tlg4020.tlg001.opp-grc1",
    "first1k_tlg1443.tlg007.1st1K-grc1",
    "first1k_tlg4021.tlg001.opp-grc1",
    "first1k_tlg1443.tlg004.1st1K-grc1",
    "first1k_tlg5026.tlg007.First1K-grc1",
    "first1k_tlg0732.tlg008.opp-grc1",
    "first1k_tlg1443.tlg006.1st1K-grc1",
    "first1k_tlg2036.tlg001.opp-grc1",
    "first1k_tlg2000.tlg001.opp-grc2",
    "first1k_tlg1799.tlg008.First1K-grc1",
    "first1k_tlg2042.tlg006.opp-grc1",
    "first1k_tlg0732.tlg007.opp-grc1",
    "first1k_tlg1443.tlg009.1st1K-grc1",
    "first1k_tlg1443.tlg010.1st1K-grc1",
    "first1k_tlg0544.tlg002.opp-grc1",
    "first1k_tlg4031.tlg002.opp-grc1",
    "first1k_tlg1443.tlg005.1st1K-grc1",
    "first1k_tlg3135.tlg001.opp-grc1",
    "first1k_tlg0645.tlg001.opp-grc1",
    "first1k_tlg0096.tlg002.First1K-grc1",
    "first1k_tlg4020.tlg002.opp-grc1",
    "first1k_tlg1443.tlg008.1st1K-grc1",
    "first1k_tlg0732.tlg005.opp-grc1",
    "first1k_tlg0015.tlg001.opp-grc1",
    "first1k_tlg2041.tlg001.opp-grc1"
]

cleaned_document_ids = remove_prefix_from_ids(document_ids)

# Initialize a dictionary to hold renaming histories
rename_histories = {doc_id: set() for doc_id in cleaned_document_ids}

# Regular expression pattern to match renaming lines
rename_pattern = re.compile(r'(.+?)\s*->\s*(.+)')

# Path to renames.txt
renames_file = "/home/gnosis/Documents/au_work/main/comp_antiquity/dat/greek/raw_data/First1KGreek/.git/filter-repo/analysis/renames.txt"

# Read and process the renames.txt file
with open(renames_file, 'r', encoding='utf-8') as f:
    for line in f:
        match = rename_pattern.match(line.strip())
        if match:
            old_name, new_name = match.groups()
            # Remove file extensions for matching
            old_base = re.sub(r'\.[^.]+$', '', old_name)
            new_base = re.sub(r'\.[^.]+$', '', new_name)
            for doc_id in cleaned_document_ids:
                if doc_id in old_base or doc_id in new_base:
                    rename_histories[doc_id].update([old_name, new_name])

# Output the renaming history for each document_id
for doc_id, paths in rename_histories.items():
    print(f"{doc_id}:")
    if paths:
        for path in sorted(paths):
            print(f"  {path}")
    else:
        print("  No renaming history found.")
