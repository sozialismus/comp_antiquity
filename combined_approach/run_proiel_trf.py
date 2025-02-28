# trf model crashes, memory spikes
# import spacy
# import json
# import os

# nlp = spacy.load("grc_proiel_trf")

# input_folder = "/home/gnosis/Documents/au_work/main/corpora/trf_ner_v2/"
# output_folder = "/home/gnosis/Documents/au_work/main/corpora/trf_ner_v2_results/"

# os.makedirs(output_folder, exist_ok=True)

# # Collect text files to process
# text_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

# # Process text files in batches
# for filename in text_files:
#     file_path = os.path.join(input_folder, filename)
#     output_path = os.path.join(output_folder, filename.replace(".txt", "_trf.jsonl"))


# def read_files():
#     """Generator function to stream text files without loading them all at once."""
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".txt"):
#             file_path = os.path.join(input_folder, filename)
#             with open(file_path, "r", encoding="utf-8") as f:
#                 yield f.read(), filename  # ✅ Yield text and filename (to keep track)

# # 🔥 Process files in batches & write results incrementally
# for doc, filename in nlp.pipe(read_files(), batch_size=1, as_tuples=True):
#     output_path = os.path.join(output_folder, filename.replace(".txt", "_trf.json"))
    
#     with open(output_path, "w", encoding="utf-8") as out_f:
#         for token in doc:
#             head = token.head.i + 1 if token.head != token else 0  # 1-based index; ROOT=0
#             dep = token.dep_  
#             deps = f"{head}:{dep}" if head > 0 else "_"  

#             token_data = {
#                 "id": token.i + 1,  
#                 "text": token.text,  
#                 "lemma": token.lemma_,  
#                 "upos": token.pos_,  
#                 "xpos": token.tag_,  
#                 "feats": str(token.morph) if token.morph else "_",  
#                 "head": head,  
#                 "dep": dep,  
#                 "deps": deps,  
#                 "misc": "_"
#             }

#             out_f.write(json.dumps(token_data, ensure_ascii=False) + "\n")  # ✅ NDJSON format
# print("✅ proiel_trf processing completed.")

# current script cannot handle large scale files, python process is being killed automatically
# import os
# import spacy
# import json
# import gc  # 🚀 Garbage collector to free memory

# # Load full spaCy model (without disabling components)
# nlp = spacy.load("grc_proiel_trf")

# # input_folder = "/home/gnosis/Documents/au_work/main/corpora/trf_ner_v2/"
# # output_folder = "/home/gnosis/Documents/au_work/main/corpora/trf_ner_v2_results/"

# # ✅ Define base directory where datasets are stored
# BASE_DIR = "/home/gnosis/Documents/au_work/main/corpora/extract/nlp"

# # ✅ Recursively find all datasets in subdirectories
# AVAILABLE_DATASETS = {}
# for root, dirs, _ in os.walk(BASE_DIR):
#     for d in dirs:
#         dataset_path = os.path.join(root, d)
#         AVAILABLE_DATASETS[d] = {
#             "input": dataset_path,  # Use detected subdirectory as input
#             "output": f"{dataset_path}_analysis"  # Append _analysis for output
#         }

# # 🏆 Step 1: Let the user pick a dataset
# if not AVAILABLE_DATASETS:
#     print("❌ No datasets found in the base directory!")
#     exit(1)

# print("\n📂 Available datasets:")
# for i, dataset in enumerate(AVAILABLE_DATASETS.keys(), 1):
#     print(f"{i}. {dataset}")

# # 🏆 Step 2: Get user selection
# try:
#     choice = int(input("\n🔹 Select a dataset (number): ").strip()) - 1
#     selected_dataset = list(AVAILABLE_DATASETS.keys())[choice]
# except (ValueError, IndexError):
#     print("⚠️ Invalid choice! Defaulting to first dataset.")
#     selected_dataset = list(AVAILABLE_DATASETS.keys())[0]

# # 🏆 Step 3: Set input and output folders dynamically
# input_folder = AVAILABLE_DATASETS[selected_dataset]["input"]
# output_folder = AVAILABLE_DATASETS[selected_dataset]["output"]

# # 🏆 Step 4: Ensure the output directory exists
# os.makedirs(output_folder, exist_ok=True)

# # 🔥 Debugging: Show selected paths
# print(f"\n✅ Selected dataset: {selected_dataset}")
# print(f"📥 Input Folder: {input_folder}")
# print(f"📤 Output Folder: {output_folder}\n")


# os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

# def read_files():
#     """Generator function to stream text files with a 2000-word limit."""
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".txt"):
#             file_path = os.path.join(input_folder, filename)
#             with open(file_path, "r", encoding="utf-8") as f:
#                 # full text, no word limit
#                 yield f.read(), filename  # ✅ No word limit (full text)
#                 # limit to reading texts to the first 2000 words
#                 # text = f.read().split()[:2000]  # ✅ Limit to first 2000 words
#                 # yield " ".join(text), filename  # ✅ Yield truncated text & filename

# # 🔥 Process files one by one (batch_size=1) to avoid memory spikes
# for doc, filename in nlp.pipe(read_files(), batch_size=1, as_tuples=True):
#     output_path = os.path.join(output_folder, filename.replace(".txt", "_trf.json"))

#     with open(output_path, "w", encoding="utf-8") as out_f:
#         for token in doc:
#             head = token.head.i + 1 if token.head != token else 0  # 1-based index; ROOT=0
#             dep = token.dep_
#             deps = f"{head}:{dep}" if head > 0 else "_"

#             token_data = {
#                 "id": token.i + 1,
#                 "text": token.text,
#                 "lemma": token.lemma_,
#                 "upos": token.pos_,
#                 "xpos": token.tag_,
#                 "feats": str(token.morph) if token.morph else "_",
#                 "head": head,
#                 "dep": dep,
#                 "deps": deps,
#                 "misc": "_"
#             }
#             out_f.write(json.dumps(token_data, ensure_ascii=False) + "\n")  # ✅ NDJSON format

#     # ✅ Free up memory after processing each file
#     del doc
#     gc.collect()  # 🚀 Force garbage collection to prevent memory buildup

# print("✅ proiel_trf processing completed.")
# updated code
import os
import spacy
import json
import gc

# ✅ Load full model (keeping all features)
nlp = spacy.load("grc_proiel_trf")

# 🏆 Dynamically select dataset paths
BASE_DIR = "/home/gnosis/Documents/au_work/main/corpora/extract/nlp"
AVAILABLE_DATASETS = {
    d: {
        "input": os.path.join(root, d),
        "output": os.path.join(root, f"{d}_analysis")
    }
    for root, dirs, _ in os.walk(BASE_DIR) for d in dirs
}

if not AVAILABLE_DATASETS:
    print("❌ No datasets found!")
    exit(1)

# 🏆 User selects dataset
print("\n📂 Available datasets:")
for i, dataset in enumerate(AVAILABLE_DATASETS.keys(), 1):
    print(f"{i}. {dataset}")

try:
    choice = int(input("\n🔹 Select a dataset (number): ").strip()) - 1
    selected_dataset = list(AVAILABLE_DATASETS.keys())[choice]
except (ValueError, IndexError):
    print("⚠️ Invalid choice! Defaulting to first dataset.")
    selected_dataset = list(AVAILABLE_DATASETS.keys())[0]

input_folder = AVAILABLE_DATASETS[selected_dataset]["input"]
output_folder = AVAILABLE_DATASETS[selected_dataset]["output"]
os.makedirs(output_folder, exist_ok=True)

print(f"\n✅ Selected dataset: {selected_dataset}")
print(f"📥 Input: {input_folder}")
print(f"📤 Output: {output_folder}\n")

def stream_sentences(filepath):
    """Stream file content sentence by sentence."""
    with open(filepath, "r", encoding="utf-8") as f:
        buffer = ""
        for line in f:
            buffer += line.strip() + " "  # Maintain sentence context
            if line.strip().endswith("."):  # Sentence boundary (Greek may differ)
                yield buffer.strip()
                buffer = ""  # Reset buffer for next sentence
        if buffer:  # Last sentence if no final period
            yield buffer.strip()

# 🔥 Process each file (avoid large memory loads)
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".txt", "_trf.json"))

        # ✅ Stream text in sentences & process in `nlp.pipe()`
        docs = nlp.pipe(stream_sentences(file_path), batch_size=10)  # Adjust batch size if needed

        # ✅ Collect results before writing (batch writes are faster)
        tokens_data = []
        for doc in docs:
            for token in doc:
                head = token.head.i + 1 if token.head != token else 0
                dep = token.dep_
                deps = f"{head}:{dep}" if head > 0 else "_"

                tokens_data.append({
                    "id": token.i + 1,
                    "text": token.text,
                    "lemma": token.lemma_,
                    "upos": token.pos_,
                    "xpos": token.tag_,
                    "feats": str(token.morph) if token.morph else "_",
                    "head": head,
                    "dep": dep,
                    "deps": deps,
                    "misc": "_"
                })

        # ✅ Write all tokens in one batch (efficient)
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(tokens_data, out_f, ensure_ascii=False, indent=2)

        # ✅ Free up memory
        del tokens_data
        gc.collect()

print("✅ proiel_trf processing completed.")
