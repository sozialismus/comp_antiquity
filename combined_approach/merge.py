import json
import os
import pandas as pd

input_folder = "/home/gnosis/Documents/au_work/main/corpus_10_results"
output_folder = "/home/gnosis/Documents/au_work/main/corpus_10_results_conllu"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith("_trf.jsonl"):
        base_name = filename.replace("_trf.jsonl", "")
        trf_path = os.path.join(input_folder, filename)
        ner_path = os.path.join(input_folder, base_name + "_ner.jsonl")

        # 📌 Define output CoNLL-U path
        conllu_filename = os.path.join(output_folder, base_name + ".conllu")

        # Check if both TRF and NER files exist
        if not os.path.exists(ner_path):
            print(f"⚠️ Warning: NER file missing for {base_name}, skipping merge.")
            continue

        with open(trf_path, "r", encoding="utf-8") as trf_file, \
             open(ner_path, "r", encoding="utf-8") as ner_file, \
             open(conllu_filename, "w", encoding="utf-8") as conllu_file:

            conllu_file.write("# This file follows Universal Dependencies format\n\n")

            for trf_line, ner_line in zip(trf_file, ner_file):
                trf_data = json.loads(trf_line)
                ner_data = json.loads(ner_line)

                # ✅ Extract token attributes safely (use "_" as default for missing values)
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
