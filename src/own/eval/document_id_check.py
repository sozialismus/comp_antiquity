import pandas as pd

# Load the master index and the local index
master_index = pd.read_csv("/home/gnosis/Documents/au_work/forsk/master.csv")
local_index = pd.read_csv("/home/gnosis/Documents/au_work/main/comp_antiquity/dat/greek/parsed_data/index.csv")

# Merge to find matching document_ids
merged = master_index.merge(local_index, on="document_id", how="left", indicator=True)

# Identify missing document_ids
missing = merged[merged["_merge"] == "left_only"]

# Report
if missing.empty:
    print("✅ All document_ids from the master index are present in the local index.")
else:
    missing_ids = missing["document_id"].to_list()
    print("❌ The following document_ids from the master index are missing in the local index:")
    print(missing_ids)

    # Extract full rows with all metadata for missing entries
    missing_full_metadata = master_index[master_index["document_id"].isin(missing_ids)]

    # Save to CSV
    output_path = "/home/gnosis/Documents/au_work/forsk/missing_documents_metadata.csv"
    missing_full_metadata.to_csv(output_path, index=False)
    print(f"📄 Metadata for missing document_ids saved to: {output_path}")
