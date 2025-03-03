import os
import spacy
import json
import gc
import sys

# Use command-line argument if provided, otherwise default to first dataset
choice = sys.argv[1] if len(sys.argv) > 1 else "1"

# ✅ Load full model
nlp = spacy.load("grc_proiel_trf")
nlp.enable_pipe("parser")  # Ensure parser is enabled

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

choice = sys.argv[1] if len(sys.argv) > 1 else "1"
# choice = input("\n🔹 Select a dataset (number): ").strip()
if not choice.isdigit():
    print("⚠️ Invalid choice! Defaulting to first dataset.")
    choice = "1"
selected_dataset = list(AVAILABLE_DATASETS.keys())[int(choice) - 1]

input_folder = AVAILABLE_DATASETS[selected_dataset]["input"]
output_folder = AVAILABLE_DATASETS[selected_dataset]["output"]
os.makedirs(output_folder, exist_ok=True)

print(f"\n✅ Selected dataset: {selected_dataset}")
print(f"📥 Input: {input_folder}")
print(f"📤 Output: {output_folder}\n")


# ✅ Function to process sentences incrementally
def process_text(file_path, output_path, doc_name):
    global_id = 1  # ✅ Unique token ID across entire file

    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # ✅ Process sentences one-by-one (generator, avoiding memory spikes)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for sent in nlp(full_text).sents:
            sentence_text = sent.text

            # ✅ Process sentence with nlp.pipe() for efficiency
            for token in nlp(sentence_text):
                if token.text.strip():  # ✅ Ignore empty tokens
                    token_data = {
                        "doc_name": doc_name,  # ✅ New Metadata
                        "global_id": global_id,  # ✅ Global token counter
                        "sentence_local_id": token.i + 1,  # ✅ Local ID within sentence
                        "text": token.text,
                        "lemma": token.lemma_,
                        "upos": token.pos_,
                        "xpos": token.tag_,
                        "feats": str(token.morph) if token.morph else "_",
                        "head": token.head.i + 1 if token.head != token else 0,
                        "dep": token.dep_,
                        "deps": f"{token.head.i + 1}:{token.dep_}" if token.head != token else "_",
                        "misc": "_"
                    }

                    out_f.write(json.dumps(token_data, ensure_ascii=False) + "\n")
                    global_id += 1  # ✅ Increment only when writing token

            # ✅ Immediately release processed Spacy objects
            del sent
            gc.collect()


# ✅ Process each file efficiently
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".txt", "_trf.json"))

        print(f"📄 Processing: {filename} ...")
        process_text(file_path, output_path, filename)

print("✅ proiel_trf processing completed.")
