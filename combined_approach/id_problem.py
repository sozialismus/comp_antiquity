import os
import spacy
import json
import gc

# ✅ Load full model (keeping all features)
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

choice = input("\n🔹 Select a dataset (number): ").strip()
if not choice.isdigit():
    print("⚠️ Invalid choice! Defaulting to first dataset.")
    choice = "1"  # Default to first
selected_dataset = list(AVAILABLE_DATASETS.keys())[int(choice) - 1]

input_folder = AVAILABLE_DATASETS[selected_dataset]["input"]
output_folder = AVAILABLE_DATASETS[selected_dataset]["output"]
os.makedirs(output_folder, exist_ok=True)

print(f"\n✅ Selected dataset: {selected_dataset}")
print(f"📥 Input: {input_folder}")
print(f"📤 Output: {output_folder}\n")

# 🔥 Process each file efficiently
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".txt", "_trf.json"))

        global_id = 1  # ✅ Global counter for token IDs
        tokens_data = []  # ✅ Store only processed results, not raw text

        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()  # ✅ Read full document (keeps context)

        # ✅ Process text with NLP model
        doc = nlp(full_text)

        for sent in doc.sents:
            # print("📜 Sentence:", sent.text)  # ✅ Debug sentence segmentation

            for token in sent:
                if token.text.strip() == "":  # ✅ Ignore empty tokens
                    continue

                head = token.head.i + 1 if token.head != token else 0
                dep = token.dep_
                deps = f"{head}:{dep}" if head > 0 else "_"

                # ✅ Store token inside loop
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

        # ✅ Write to NDJSON format (one JSON object per line)
        with open(output_path, "w", encoding="utf-8") as out_f:
            for token_data in tokens_data:
                out_f.write(json.dumps(token_data, ensure_ascii=False) + "\n")

        # ✅ Free up memory
        del doc, tokens_data
        gc.collect()

print("✅ proiel_trf processing completed.")
