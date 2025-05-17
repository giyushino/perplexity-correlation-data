#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ~/miniconda3/etc/profile.d/conda.sh
conda activate perplexity
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc-per-node=8 train.py


