"""Script responsible for parsing the corpus,
producing log and metadata files"""
# Importing packages
import os
import glob

import pandas as pd
from pathlib import Path

from utils.parsing import (
    process_files,
    PerseusParser,
    # SEPAParser,
    PseudepigraphaParser,
    AttalusScrapedTextParser,
    SBLGNTParser,
)

# setting cwd
os.chdir('/home/gnosis/Documents/au_work/main/comp_antiquity/')

RAW_PATH = "dat/greek/raw_data"
OUT_PATH = "dat/greek/parsed_data/"


def main() -> None:
    print(
        "--------------------------\n"
        "------CORPUS PARSER-------\n"
        "--------------------------"
    )
    print("Collecting file paths.")
    # Creating output path
    Path(OUT_PATH).mkdir(exist_ok=True, parents=True)

    # Producing paths for raw files
    perseus_path = os.path.join(RAW_PATH, "canonical-greekLit/data")
    first1k_path = os.path.join(RAW_PATH, "First1KGreek/data")
    # sepa_path = os.path.join(RAW_PATH, "SEPA")
    pseudepigrapha_path = os.path.join(
        RAW_PATH, "Online-Critical-Pseudepigrapha/static/docs"
    )
    sblgnt_path = os.path.join(RAW_PATH, "SBLGNT-master/data/sblgnt/xml")
    attalus_path = os.path.join(RAW_PATH, "attalus/")
    

    # Getting all exact file paths
    perseus_files = glob.glob(
        os.path.join(perseus_path, "**/*grc*.xml"), recursive=True
    )
    first1k_files = glob.glob(
        os.path.join(first1k_path, "**/*grc*.xml"), recursive=True
    )
    # sepa_files = glob.glob(os.path.join(sepa_path, "**/*.usx"), recursive=True)
    pseudepigrapha_files = glob.glob(
        os.path.join(pseudepigrapha_path, "*.xml")
    )
    sblgnt_files = glob.glob(os.path.join(sblgnt_path, "*.xml"))
    attalus_files = glob.glob(os.path.join(attalus_path, "*.txt"))
    
    print("Processing corpora:")
    print(" - Perseus")
    process_files(
        paths=perseus_files,
        parser=PerseusParser(),
        source_name="perseus",
        dest=OUT_PATH,
    )

    print(" - First1KGreek")
    # Processing first1k files
    process_files(
        paths=first1k_files,
        parser=PerseusParser(),
        source_name="first1k",
        dest=OUT_PATH,
    )

    # print(" - SEPA")
    # process_files(
    #     paths=sepa_files,
    #     parser=SEPAParser(),
    #     source_name="SEPA",
    #     dest=OUT_PATH,
    # )

    print(" - Pseudepigrapha")
    process_files(
        paths=pseudepigrapha_files,
        parser=PseudepigraphaParser(),
        source_name="pseudepigrapha",
        dest=OUT_PATH,
    )

    print(" - SBLGNT")
    process_files(
        paths=sblgnt_files,
        parser=SBLGNTParser(),  # Use the new parser or an existing one
        source_name="SBLGNT",
        dest=OUT_PATH,
    )

    print(" - Attalus")
    process_files(
        paths=attalus_files,
        parser=AttalusScrapedTextParser(),
        source_name="attalus",
        dest=OUT_PATH,
    )

    # Aggregating index files
    print("Aggregating indices")
    index_files = glob.glob(os.path.join(OUT_PATH, "**/index.csv"))
    index_dfs = [pd.read_csv(file, index_col=0) for file in index_files]
    index_dat = pd.concat(index_dfs, ignore_index=True)

    # Saving index file
    index_dat.to_csv(os.path.join(OUT_PATH, "index.csv"))
    print("DONE")


if __name__ == "__main__":
    main()
