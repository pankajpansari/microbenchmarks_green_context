#!/bin/bash

export WORKSPACE=/workspace
export VENV="$WORKSPACE/venv"
export HF_HOME="$WORKSPACE/.cache/huggingface"

mkdir -p $HF_HOME

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment at $VENV"
    python -m venv $VENV
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -r $WORKSPACE/msr_talk/requirements.txt 
else
    echo "Virtual environment already exists at $VENV"
fi

source "$VENV/bin/activate"

apt update && apt install vim -yv

git config --global user.name "pankajpansari"
git config --global user.email "pankaj.pansari1@gmail.com"
git config --global credential.helper "store --file=$WORKSPACE/.git-credentials"
