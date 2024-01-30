#!/bin/bash
# Used for training TSA-AP model
set -e
anomaly_type="trend" # platform, trend, extremum, mean for phsyionet or a, b for mocap
a="-0.2-rand-rand" # suffix for wandb logging
dataset="physionet" # physionet or mocap

config_path="configs/tsap/${dataset}_${anomaly_type}_tsap.yml"
monitor="zval_loss"
ckpt_name="${dataset}_tsap_a=${a_size}_${anomaly_type}_a=$a"
results_path="results/$ckpt_name"
python main.py \
    --ckpt_name "$ckpt_name" \
    --ckpt_monitor "$monitor" \
    --config_path "$config_path" \
    --results_path "$results_path" \
    --train_mode \
    #--wandb 