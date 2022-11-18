#!/bin/bash
set -x
echo "=======Creating virtual env========="
python3 -m virtualenv  .venv_seal
source .venv_seal/bin/activate

echo "=======Install test requirements======="
pip install -r test_requirements.txt

echo "=======Install core requirements======"
pip install -r core_requirements.txt
# This is temporary code to handle nltk error.
python3 resolve_nltk_err.py

echo "========Create log dir for single runs========="
mkdir -p logs

echo "=======Login to wandb (optional)==============="
wandb init

echo "Do 'source .venv_seal/bin/activate' to load the enviroment."
