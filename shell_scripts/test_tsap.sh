#!/bin/bash
# Used for testing the TSA-AP model
set -e
anomaly_type="trend" # platform, trend, extremum, mean for phsyionet or a, b for mocap
dataset="physionet" # physionet or mocap

config_path="configs/tsap/${dataset}_${anomaly_type}_tsap.yml"
load_from_ckpt= ".." # path to the checkpoint to load TSAP from
results_path="results/"
python main.py \
    --config_path "$config_path" \
    --results_path "$results_path" \
    --test_mode \
    --load_from_ckpt "$load_from_ckpt" 