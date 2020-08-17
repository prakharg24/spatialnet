#!/bin/bash
#
# Create new conda environment

conda create -n spatialnet python=3.6 -y

conda activate spatialnet

# Install pytorch
conda install -c pytorch pytorch -y

# Install other dependencies
conda install attrs -y
conda install numpy -y
conda install torchvision -y
conda install pillow -y
conda install tqdm -y

# Exit environment
conda deactivate
