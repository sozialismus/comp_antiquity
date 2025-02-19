import spacy
import json
import os

nlp = spacy.load("grc_ner_trf")

input_folder = "/home/gnosis/Documents/au_work/main/corpus_10"
output_folder = "/home/gnosis/Documents/au_work/main/corpus_10_results"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 🔍 Process text with the NER model
        doc = nlp(text)

        # 🛠 Extract entity information
        ner_data = {
            "tokens": [token.text for token in doc],
            "ents": [
                {"start": ent.start, "end": ent.end, "label": ent.label_}
                for ent in doc.ents
            ]
        }

        # 📌 Save NER results
        output_path = os.path.join(output_folder, filename.replace(".txt", "_ner.json"))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ner_data, f, ensure_ascii=False, indent=4)

print("✅ NER processing completed.")
