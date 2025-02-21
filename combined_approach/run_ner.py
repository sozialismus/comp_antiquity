import spacy
import json
import os

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
                yield f.read(), filename  # Yield text + filename instead of storing in a list

# Process the files with nlp.pipe() in batches of 5 files at a time
for doc, filename in nlp.pipe(read_files(), batch_size=5, as_tuples=True):
    # Process the NER model for the text of each file
    doc = nlp(doc)  # Apply NER to the document

    # Extract entity information for each document
    ner_data = {
        "tokens": [token.text for token in doc],
        "ents": [
            {"start": ent.start, "end": ent.end, "label": ent.label_}
            for ent in doc.ents
        ]
    }

    # Save the NER results to disk immediately to avoid memory spikes
    output_path = os.path.join(output_folder, filename.replace(".txt", "_ner.json"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ner_data, f, ensure_ascii=False, indent=4)

print("✅ NER processing completed.")
