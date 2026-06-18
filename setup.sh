#!/bin/bash

export WORKSPACE=/workspace
export VENV="$WORKSPACE/venv"
export REPO="$WORKSPACE/microbenchmarks_green_context"
export HF_HOME="$WORKSPACE/.cache/huggingface"

mkdir -p $HF_HOME

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment at $VENV"
    python3 -m venv $VENV
    
    source "$VENV/bin/activate"
    
    pip install --upgrade pip uv
    uv pip install --upgrade pip
    uv pip install -r "$REPO/requirements.txt"
    uv pip install torch --index-url https://download.pytorch.org/whl/cu124
    uv pip install flashinfer-python flashinfer-cubin
    uv pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu124
else
    echo "Virtual environment already exists at $VENV"
    source "$VENV/bin/activate"
fi

git config --global user.name "pankajpansari"
git config --global user.email "pankaj.pansari1@gmail.com"
git config --global credential.helper "store --file=$WORKSPACE/.git-credentials"
