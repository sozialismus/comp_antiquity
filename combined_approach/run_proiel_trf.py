import spacy
import json
import os

nlp = spacy.load("grc_proiel_trf")

input_folder = "/home/gnosis/Documents/au_work/main/corpus_10"
output_folder = "/home/gnosis/Documents/au_work/main/corpus_10_results"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        doc = nlp(text)

        # Save dependency parsing results
        output_path = os.path.join(output_folder, filename.replace(".txt", "_trf.json"))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc.to_json(), f, ensure_ascii=False, indent=4)

print("proiel_trf processing completed.")
