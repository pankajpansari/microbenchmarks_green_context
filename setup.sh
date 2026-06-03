#!/bin/bash

export WORKSPACE=/workspace
export VENV="$WORKSPACE/venv"
export HF_HOME="$WORKSPACE/.cache/huggingface"

mkdir -p $HF_HOME

python3 -m pip install --upgrade pip

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment at $VENV"
    python3 -m venv $VENV
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -r $WORKSPACE/requirements.txt
    "$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu124
    "$VENV/bin/pip" install flashinfer-python flashinfer-cubin
    "$VENV/bin/pip" install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu124
else
    echo "Virtual environment already exists at $VENV"
fi

source "$VENV/bin/activate"

git config --global user.name "pankajpansari"
git config --global user.email "pankaj.pansari1@gmail.com"
git config --global credential.helper "store --file=$WORKSPACE/.git-credentials"
