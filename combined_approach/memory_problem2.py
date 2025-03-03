import os
import spacy
import json
import gc

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

choice = input("\n🔹 Select a dataset (number): ").strip()
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

    with open(file_path, "r", encoding="utf-8") as f, open(output_path, "w", encoding="utf-8") as out_f:
        text_chunks = []

        # ✅ Read and accumulate small chunks of text
        for line in f:
            line = line.strip()
            if line:
                text_chunks.append(line)

            # ✅ Process when enough text has accumulated (avoids full memory load)
            if len(text_chunks) >= 10:  # Tune this batch size if needed
                for doc in nlp.pipe(text_chunks, batch_size=10):
                    for sent in doc.sents:  # ✅ Correct sentence splitting
                        global_id = process_sentence(sent, out_f, global_id, doc_name)
                
                text_chunks = []  # ✅ Clear memory

        # ✅ Process remaining sentences
        if text_chunks:
            for doc in nlp.pipe(text_chunks, batch_size=10):
                for sent in doc.sents:
                    global_id = process_sentence(sent, out_f, global_id, doc_name)

# ✅ Function to process a single sentence
def process_sentence(sent, out_f, global_id, doc_name):
    sentence_tokens = []
    
    for token in sent:
        if token.text.strip() == "":  # ✅ Ignore empty tokens
            continue

        head = token.head.i + 1 if token.head != token else 0
        dep = token.dep_
        deps = f"{head}:{dep}" if head > 0 else "_"

        # ✅ Store token information
        token_data = {
            "global_id": global_id,  # ✅ Unique ID for entire document
            "id": token.i + 1,  # ✅ Token ID within sentence
            "text": token.text,
            "lemma": token.lemma_,
            "upos": token.pos_,
            "xpos": token.tag_,
            "feats": str(token.morph) if token.morph else "_",
            "head": head,
            "dep": dep,
            "deps": deps,
            "misc": "_",
            "doc_name": doc_name  # ✅ Document name for metadata
        }
        
        out_f.write(json.dumps(token_data, ensure_ascii=False) + "\n")
        global_id += 1  # ✅ Increment unique token ID

    return global_id

# ✅ Process each file
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".txt", "_trf.json"))
        # debug, print which document is being worked on
        print(f"📄 Processing: {filename} ...")
        process_text(file_path, output_path, filename)

print("✅ proiel_trf processing completed.")
