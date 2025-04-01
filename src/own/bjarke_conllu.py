import os
import json
import os
import json
import logging
import gc
import sys
from pathlib import Path

# Set base export directory
BASE_EXPORT_DIR = Path("dat/export/bjarke_test")
LOG_DIR = BASE_EXPORT_DIR / "logs"
LOG_FILE = LOG_DIR / "merge_process.log"

def setup_logging():
    """Sets up logging to a specific file while also printing logs to console."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)  # Ensure log directory exists

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    logging.info("Logging initialized.")

def merge_annotations_to_conllu_for_directory(source_dir):
    """Merges Proiel and NER JSON files into a single CoNLL-U formatted file for the given directory,
    outputting a continuous unsplit file (i.e. without sentence boundaries).
    """
    for id_dir in source_dir.iterdir():
        if not id_dir.is_dir():
            continue

        annotations_folder = id_dir / "annotations"
        output_file = annotations_folder / (id_dir.name + ".conllu")
        
        ner_file = annotations_folder / (id_dir.name + "-ner.json")
        proiel_file = annotations_folder / (id_dir.name + "-proiel.json")
        
        if not ner_file.exists() or not proiel_file.exists():
            logging.warning(f"Missing annotation files for {id_dir.name}")
            continue
        
        try:
            with open(ner_file, "r", encoding="utf-8") as ner_f, \
                 open(proiel_file, "r", encoding="utf-8") as proiel_f, \
                 open(output_file, "w", encoding="utf-8") as conllu_f:
                
                conllu_f.write(f"# newdoc id = {id_dir.name}\n")
                conllu_f.write("# global.features = syntax_not_annotated\n\n")
                
                # Create dictionaries keyed by token "id"
                ner_data = {entry["id"]: entry for entry in map(json.loads, ner_f)}
                proiel_data = {entry["id"]: entry for entry in map(json.loads, proiel_f)}
                
                sorted_tokens = sorted(proiel_data.keys(), key=lambda x: int(x))
                global_token_counter = 1  # Global counter for token IDs
                
                for token_id in sorted_tokens:
                    proiel_entry = proiel_data[token_id]
                    ner_entry = ner_data.get(token_id, {"ner": "O"})
                    
                    form = proiel_entry.get("text", "_")
                    lemma = proiel_entry.get("lemma", "_")
                    upos = proiel_entry.get("upos", "_")
                    xpos = proiel_entry.get("xpos", "_")
                    feats = proiel_entry.get("feats", "_")
                    head = proiel_entry.get("head", 0)
                    deprel = proiel_entry.get("dep", "_")
                    deps = proiel_entry.get("deps", "_")
                    ner_label = ner_entry.get("ner", "O")
                    
                    misc_field = f"NER={ner_label}" if ner_label != "O" else "_"
                    
                    # Write each token using the global token counter
                    conllu_row = [
                        str(global_token_counter),
                        form,
                        lemma,
                        upos,
                        xpos,
                        feats,
                        str(head),
                        deprel,
                        deps,
                        misc_field
                    ]
                    conllu_f.write("\t".join(conllu_row) + "\n")
                    global_token_counter += 1
                
                # End the file with a newline.
                conllu_f.write("\n")
            
            logging.info(f"Merged {ner_file} and {proiel_file} into {output_file}")
        except Exception as e:
            logging.error(f"Error merging annotations for {id_dir.name}: {e}")
            continue

def merge_annotations_to_conllu():
    """Traverses all subdirectories of BASE_EXPORT_DIR and applies the merging function."""
    for source_dir in BASE_EXPORT_DIR.iterdir():
        if not source_dir.is_dir():
            continue
        merge_annotations_to_conllu_for_directory(source_dir)

if __name__ == "__main__":
    setup_logging()
    merge_annotations_to_conllu()
    print(f"Merging of NER and Proiel files to CoNLL-U format completed. Check log file: {LOG_FILE}")
