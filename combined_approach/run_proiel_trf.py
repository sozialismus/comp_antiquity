import spacy
import json
import os

nlp = spacy.load("grc_proiel_trf")

input_folder = "/home/gnosis/Documents/au_work/main/corpora/trf_ner_v2/"
output_folder = "/home/gnosis/Documents/au_work/main/corpora/trf_ner_v2_results/"

os.makedirs(output_folder, exist_ok=True)

# Collect text files to process
text_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

# Process text files in batches
for filename in text_files:
    file_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename.replace(".txt", "_trf.jsonl"))


def read_files():
    """Generator function to stream text files without loading them all at once."""
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                yield f.read(), filename  # ✅ Yield text and filename (to keep track)

# 🔥 Process files in batches & write results incrementally
for doc, filename in nlp.pipe(read_files(), batch_size=2, as_tuples=True):
    output_path = os.path.join(output_folder, filename.replace(".txt", "_trf.json"))
    
    with open(output_path, "w", encoding="utf-8") as out_f:
        for token in doc:
            head = token.head.i + 1 if token.head != token else 0  # 1-based index; ROOT=0
            dep = token.dep_  
            deps = f"{head}:{dep}" if head > 0 else "_"  

            token_data = {
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
            }

            out_f.write(json.dumps(token_data, ensure_ascii=False) + "\n")  # ✅ NDJSON format

print("✅ proiel_trf processing completed.")
