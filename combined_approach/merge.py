import json
import os
import pandas as pd
import json
import os
import pandas as pd

input_folder = "~/Documents/au_work/main/corpora/trf_ner_v2_results/"
output_folder = "~/Documents/au_work/main/corpora/trf_ner_v2_results_conllu/"

os.makedirs(output_folder, exist_ok=True)


# Collect all .jsonl files
jsonl_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]

# Find matching _trf.jsonl and _ner.jsonl files
file_pairs = {}
for filename in jsonl_files:
    base_name = filename.replace("_trf.json", "").replace("_ner.json", "")
    
    if base_name not in file_pairs:
        file_pairs[base_name] = {}
    
    if filename.endswith("_trf.json"):
        file_pairs[base_name]["trf"] = os.path.join(input_folder, filename)
    elif filename.endswith("_ner.json"):
        file_pairs[base_name]["ner"] = os.path.join(input_folder, filename)

# Merge TRF and NER files into CoNLL-U format
for base_name, paths in file_pairs.items():
    if "trf" not in paths or "ner" not in paths:
        print(f"⚠️ Warning: Missing TRF or NER file for {base_name}, skipping merge.")
        continue

    trf_path = paths["trf"]
    ner_path = paths["ner"]
    conllu_filename = os.path.join(output_folder, base_name + ".conllu")

    with open(trf_path, "r", encoding="utf-8") as trf_file, \
         open(ner_path, "r", encoding="utf-8") as ner_file, \
         open(conllu_filename, "w", encoding="utf-8") as conllu_file:

        conllu_file.write("# This file follows Universal Dependencies format\n\n")

        # Read files line-by-line (NDJSON format)
        for trf_line, ner_line in zip(trf_file, ner_file):
            trf_data = json.loads(trf_line)  # ✅ Read token data from TRF
            ner_data = json.loads(ner_line)  # ✅ Read token data from NER

            # ✅ Extract token attributes safely
            token_id = trf_data.get("id", 0)  # CoNLL-U ID (1-based index)
            form = trf_data.get("text", "_")  # Word form
            lemma = trf_data.get("lemma", "_")  # Lemma
            upos = trf_data.get("upos", "_")  # Universal POS
            xpos = "_"  # XPOS (not available)
            feats = trf_data.get("feats", "_")  # Morphological features
            head = trf_data.get("head", 0)  # Root is 0
            deprel = trf_data.get("dep", "_")  # Dependency relation
            deps = trf_data.get("deps", "_")  # Enhanced dependencies

            # 🏷️ Named Entity Label (stored in MISC only if present)
            ner_label = ner_data.get("ner", "O")
            misc_field = f"NER={ner_label}" if ner_label != "O" else "_"

            # 📌 Append token data to CoNLL-U format
            conllu_row = [token_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc_field]
            conllu_file.write("\t".join(map(str, conllu_row)) + "\n")

        conllu_file.write("\n")  # Separate sentences with a blank line

print("✅ Merged proiel_trf and NER outputs into CoNLL-U format.")
