#!/bin/bash

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html

pip install gplearn

pip install numpy

pip install jupyterlab

pip install ipywidgets

pip install tqdm

pip install matplotlib

pip install networkx

pip install einops

if ! command -v julia &> /dev/null
then
    echo "Julia could not be found, please fix this and retry."
    exit
fi

pip install pysr

python -c 'import pysr; pysr.install()'
