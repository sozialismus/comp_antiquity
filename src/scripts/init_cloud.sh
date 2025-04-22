#!/bin/bash
# initialize ucloud - clone repo and init conda envs - trying to get proiel_trf and ner_trf to function
cd /work

# Ensure Miniconda is available
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Miniconda not found, installing..."
    curl -s -L -o /tmp/miniconda_installer.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda_installer.sh -b -f -p /work/miniconda3

fi

# Source conda environment so that 'conda' command is available.
eval "$(/work/miniconda3/bin/conda shell.bash hook)"

conda init

# Create proiel trf environment (only needed once)
if ! conda env list | grep -q "proiel_trf"; then
    echo "Creating proiel_trf environment from YAML..."
    conda env create -f comp_antiquity/requirements/proiel_trf_environment.yml
fi

# Create NER environment (if applicable)
if ! conda env list | grep -q "ner"; then
    echo "Creating NER environment from YAML..."
    conda env create -f comp_antiquity/requirements/ner_environment.yml
fi

echo "All environments are set up."

# Activate the parsing environment and run a sample command
conda activate ner
echo "Running parsing task..."
# Replace with your actual command to process the corpus
conda list

# Optionally, deactivate at the end (or switch to another environment)
conda deactivate

conda info --envs

