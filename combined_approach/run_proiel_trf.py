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

        # Process text with the NLP model
        doc = nlp(text)

        # Extract relevant token attributes for UD-compliant format
        tokens_data = []
        for token in doc:
            token_data = {
                "id": token.i + 1,  # Token ID (1-based index)
                "text": token.text,  # FORM (Word form)
                "lemma": token.lemma_,  # LEMMA
                "upos": token.pos_,  # UPOS (Universal POS)
                "xpos": token.tag_,  # XPOS (Language-specific POS)
                "feats": str(token.morph) if token.morph else "_",  # FEATS (Morphological features)
                "head": token.head.i if token.head != token else 0,  # HEAD (root = 0)
                "dep": token.dep_,  # DEPREL (Dependency relation)
                "deps": "_",  # DEPS (Enhanced dependencies, not extracted here)
                "misc": "_"  # MISC (Can be used later for named entity annotation)
            }
            tokens_data.append(token_data)

        # Save dependency parsing results
        output_path = os.path.join(output_folder, filename.replace(".txt", "_trf.json"))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"tokens": tokens_data}, f, ensure_ascii=False, indent=4)

print("proiel_trf processing completed.")
