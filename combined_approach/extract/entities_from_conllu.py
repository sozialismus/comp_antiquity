import re
import csv
import json

def extract_conllu_data(conllu_file, output_format="csv"):
    """
    Extracts word ID, form, lemma, UPOS, and NER tag from a CoNLL-U file.

    Args:
        conllu_file (str): Path to the input .conllu file.
        output_format (str): "csv" or "json" for output format.

    Returns:
        List of extracted data (also writes to file).
    """

    extracted_data = []

    with open(conllu_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Skip empty lines and comments

            columns = line.split("\t")
            if len(columns) < 10:
                continue  # Skip malformed lines

            word_id, form, lemma, upos, misc = columns[0], columns[1], columns[2], columns[3], columns[9]

            # Extract NER tag from MISC column (format: "NER=TAG")
            ner_match = re.search(r"NER=([^\s|]+)", misc)
            ner = ner_match.group(1) if ner_match else "O"  # Default to "O" (outside named entities)

            extracted_data.append({
                "ID": word_id,
                "FORM": form,
                "LEMMA": lemma,
                "UPOS": upos,
                "NER": ner
            })

    # ✅ Save output based on format
    output_file = conllu_file.replace(".conllu", f".{output_format}")

    if output_format == "csv":
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["ID", "FORM", "LEMMA", "UPOS", "NER"])
            writer.writeheader()
            writer.writerows(extracted_data)
    elif output_format == "json":
        with open(output_file, "w", encoding="utf-8") as jsonfile:
            json.dump(extracted_data, jsonfile, ensure_ascii=False, indent=4)

    print(f"✅ Extracted data saved to {output_file}")
    return extracted_data

# Example Usage:
# extract_conllu_data("example.conllu", output_format="csv")  # Outputs a CSV file
# extract_conllu_data("example.conllu", output_format="json") # Outputs a JSON file
