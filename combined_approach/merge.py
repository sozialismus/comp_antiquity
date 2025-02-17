import json
import os
import pandas as pd

input_folder = "/home/gnosis/Documents/au_work/main/corpus_10_results"
output_folder = "/home/gnosis/Documents/au_work/main/corpus_10_results_conllu"
os.makedirs(output_folder, exist_ok=True)  # Corrected "true" to "True"

for filename in os.listdir(input_folder):
    if filename.endswith("_trf.json"):
        base_name = filename.replace("_trf.json", "")
        trf_path = os.path.join(input_folder, filename)
        ner_path = os.path.join(input_folder, base_name + "_ner.json")

        with open(trf_path, "r", encoding="utf-8") as f:
            trf_data = json.load(f)

        with open(ner_path, "r", encoding="utf-8") as f:
            ner_data = json.load(f)

        # Validate token structure
        if not isinstance(trf_data.get("tokens"), list):
            print(f"Warning: 'tokens' missing or not a list in {trf_path}")
            continue  # Skip this file

        # Extract named entities from NER results
        ner_entities = {}
        for ent in ner_data.get("ents", []):
            for i in range(ent["start"], ent["end"]):
                ner_entities[i] = ent["label"]

        conllu_data = []
        for i, token in enumerate(trf_data["tokens"]):
            print("Token keys:", token.keys())  # Debugging

            # Get token information safely
            text = token.get("text", token.get("orth", token.get("form", "_")))
            lemma = token.get("lemma", "_")
            upos = token.get("upos", "_")
            feats = token.get("feats", "_")
            head = token.get("head", -1) + 1 if token.get("head", -1) != i else 0
            dep = token.get("dep", "_")

            # Get NER label
            ner_label = ner_entities.get(i, None)
            misc_field = f"ner={ner_label}" if ner_label else "_"

            # Append to CoNLL-U format
            conllu_data.append([
                i + 1, text, lemma, upos, "_", feats, head, dep, "_", misc_field
            ])

        # Save CoNLL-U file
        conllu_filename = os.path.join(output_folder, base_name + ".conllu")
        with open(conllu_filename, "w", encoding="utf-8") as f:
            f.write("# This file follows Universal Dependencies format\n\n")
            for row in conllu_data:
                f.write("\t".join(map(str, row)) + "\n")
            f.write("\n")

print("CoNLL-U files saved.")
