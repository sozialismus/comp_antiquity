import spacy
import json
import os
import gc  # 🚀 Garbage collector to free memory

nlp = spacy.load("grc_ner_trf")

input_folder = "/home/gnosis/Documents/au_work/main/corpora/trf_ner_v2"
output_folder = "/home/gnosis/Documents/au_work/main/corpora/trf_ner_v2_results"

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
