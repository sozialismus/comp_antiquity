import json
import os
import pandas as pd

input_folder = "/home/gnosis/Documents/au_work/main/corpus_10_results"
output_folder = "/home/gnosis/Documents/au_work/main/corpus_10_results_conllu"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith("_trf.json"):
        base_name = filename.replace("_trf.json", "")
        trf_path = os.path.join(input_folder, filename)
        ner_path = os.path.join(input_folder, base_name + "_ner.json")

        # Load dependency parsing data (proiel_trf)
        with open(trf_path, "r", encoding="utf-8") as f:
            trf_data = json.load(f)

        # Load Named Entity Recognition (NER) results
        with open(ner_path, "r", encoding="utf-8") as f:
            ner_data = json.load(f)

        # Validate token structure
        if not isinstance(trf_data.get("tokens"), list):
            print(f"Warning: 'tokens' missing or not a list in {trf_path}")
            continue  # Skip this file

        # 📌 Extract Named Entities and store in a dictionary
        ner_entities = {}
        for ent in ner_data.get("ents", []):
            for i in range(ent["start"], ent["end"]):  # Assign NER labels to token indices
                ner_entities[i] = ent["label"]

        # 🛠 Construct CoNLL-U format data
        conllu_data = []
        for i, token in enumerate(trf_data["tokens"]):
            # ✅ Extract token attributes safely (use "_" as default for missing values)
            token_id = i + 1  # CoNLL-U ID (1-based index)
            form = token.get("text", token.get("orth", token.get("form", "_")))  # Word form
            lemma = token.get("lemma", "_")  # Lemma
            upos = token.get("upos", "_")  # Universal POS
            xpos = "_"  # XPOS (not available)
            feats = token.get("feats", "_")  # Morphological features
            head = token.get("head", -1) + 1 if token.get("head", -1) != i else 0  # Root is 0
            deprel = token.get("dep", "_")  # Dependency relation
            deps = "_"  # Enhanced dependencies (not available)

            # 🏷️ Named Entity Label (stored in MISC only if present)
            ner_label = ner_entities.get(i, None)
            misc_field = f"ner={ner_label}" if ner_label else "_"

            # 📌 Append token data to CoNLL-U format
            conllu_data.append([
                token_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc_field
            ])

        # 📝 Save CoNLL-U file
        conllu_filename = os.path.join(output_folder, base_name + ".conllu")
        with open(conllu_filename, "w", encoding="utf-8") as f:
            f.write("# This file follows Universal Dependencies format\n\n")
            for row in conllu_data:
                f.write("\t".join(map(str, row)) + "\n")
            f.write("\n")

print("✅ CoNLL-U files saved successfully.")
