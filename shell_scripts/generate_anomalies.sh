#!/bin/bash
# Used for training TSA-AP model
set -e
clean_df_path="data_store/physionet_clean/physionet_scaled.parquet"
anomaly_type="platform"
name="physionet_a"

python src/anomalies.py \
    --data_path $clean_df_path \
    --anom_type $anomaly_type \
    --name $name