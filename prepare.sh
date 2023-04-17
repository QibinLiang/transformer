#!/bin/bash

# stage -1: download the data
# stage 0: preprocess the data
# stage 1: train the model

stage=-1
stop_stage=1
token_level="char"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Download the data"
    python utils/download.py
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Preprocess the data"
    python utils/preprocessing.py --token_level "$token_level"
fi