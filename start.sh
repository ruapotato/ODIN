#!/bin/bash
source ./pyenv/bin/activate
export LD_LIBRARY_PATH=/home/hamner/ODIN/pyenv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
python ./main.py
