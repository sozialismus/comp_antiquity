import spacy
import json
import os
import gc  # 🚀 Garbage collector to free memory

nlp = spacy.load("grc_ner_trf")

# input_folder = "/home/gnosis/Documents/au_work/main/corpora/trf_ner_v2"
# output_folder = "/home/gnosis/Documents/au_work/main/corpora/trf_ner_v2_results"

# ✅ Define base directory where datasets are stored
BASE_DIR = "/home/gnosis/Documents/au_work/main/corpora/extract/nlp"

# ✅ Recursively find all datasets in subdirectories
AVAILABLE_DATASETS = {}
for root, dirs, _ in os.walk(BASE_DIR):
    for d in dirs:
        dataset_path = os.path.join(root, d)
        AVAILABLE_DATASETS[d] = {
            "input": dataset_path,  # Use detected subdirectory as input
            "output": f"{dataset_path}_analysis"  # Append _analysis for output
        }

# 🏆 Step 1: Let the user pick a dataset
if not AVAILABLE_DATASETS:
    print("❌ No datasets found in the base directory!")
    exit(1)
print("\n📂 Available datasets:")
dataset_list = list(AVAILABLE_DATASETS.keys())  # Convert to list for indexing

for i, dataset in enumerate(dataset_list, 1):  # Start numbering from 1
    print(f"{i}. {dataset}")

# ✅ Step 2: Let the user pick a dataset
try:
    choice = int(input("\n🔹 Select a dataset (number): ").strip())  
    if 1 <= choice <= len(dataset_list):
        selected_dataset = dataset_list[choice - 1]  # Convert to zero-based index
    else:
        raise ValueError
except (ValueError, IndexError):
    print("⚠️ Invalid choice! Defaulting to the first dataset.")
    selected_dataset = dataset_list[0]  # Default to first dataset
# 🏆 Step 3: Set input and output folders dynamically
input_folder = AVAILABLE_DATASETS[selected_dataset]["input"]
output_folder = AVAILABLE_DATASETS[selected_dataset]["output"]

# 🏆 Step 4: Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# 🔥 Debugging: Show selected paths
print(f"\n✅ Selected dataset: {selected_dataset}")
print(f"📥 Input Folder: {input_folder}")
print(f"📤 Output Folder: {output_folder}\n")

os.makedirs(output_folder, exist_ok=True)

# Open all input files as a generator to reduce memory footprint
def read_files():
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                # full text, no word limit
                # yield f.read(), filename  # ✅ No word limit (full text)
                # limit to reading texts to the first 2000 words
                text = f.read().split()[:2000]  # ✅ Limit to first 2000 words
                yield " ".join(text), filename  # ✅ Yield truncated text & filename

# Process the files with nlp.pipe() in batches of x files at a time
for doc, filename in nlp.pipe(read_files(), batch_size=1, as_tuples=True):
    output_path = os.path.join(output_folder, filename.replace(".txt", "_ner.json"))

    with open(output_path, "w", encoding="utf-8") as f:
        for token in doc:
            token_data = {
                "id": token.i + 1,  # ✅ 1-based token index
                "text": token.text,  # ✅ Token text
                "ner": token.ent_type_ if token.ent_type_ else "O"  # ✅ Named Entity Label (O if none)
            }
            f.write(json.dumps(token_data, ensure_ascii=False) + "\n")  # ✅ Write NDJSON format

    # Free memory after processing each file
    del doc
    gc.collect()

print("✅ NER processing completed (NDJSON format).")
