#!/bin/bash
# initialize ucloud - clone repo and init conda envs - trying to get proiel_trf and ner_trf to function
cd /work

# move conda environment & comp_antiquity
# mv /work/cleaning_texts/miniconda3 /work/
cp -r /work/cleaning_texts/work_backup_23_04/comp_antiquity/ /work/

# Source conda environment so that 'conda' command is available.
# eval "$(/work/miniconda3/bin/conda shell.bash hook)"
eval "$(/work/miniconda3/bin/conda shell.bash hook)"

conda init

# Replace with your actual command to process the corpus
conda list

echo "All environments are set up."

echo "Testing activation"
# Activate the parsing environment and run a sample command
conda activate ner

# Optionally, deactivate at the end (or switch to another environment)
conda deactivate

conda info --envs

# Keep the shell open
#tail -f /dev/null
exec bash

