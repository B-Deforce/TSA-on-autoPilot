## Description
This repository contains the code for *TSA on AutoPilot: Self-tuning Self-supervised Time Series Anomaly Detection*. 

## Data
PhysioNet A-G and Mocap A-B are available [here](https://drive.google.com/drive/folders/1GtQ9Iahtl9SamL0PZpSxoZwhrvXdFGmD?usp=sharing)

## Training and Testing TSAP
To train TSAP, specify the configuration file in `configs/tsap` and adapt `shell_scripts/train_tsap.sh` accordingly. To test TSAP, specify the configuration file in `configs/tsap` and adapt `shell_scripts/test_tsap.sh` accordingly.
```bash
# Train TSAP on PhysioNet C
sh shell_scripts/train_tsap.sh
```
Three key parameters in the configuration files are `aug_params`, `a_init`, and `anom_data_path`. `aug_params` specifies which hyperparameters to learn and which ones to randomize (`lvl`, `loc`, `len`). `a_init` specifies the initial values of the augmentation hyperparameter $\mathbb{a}$. `anom_data_path` specifies the path to the anomalous data (i.e. PhysioNet A-G or Mocap A-B).

## File Structure

The project is organized as follows:

- **/checkpoints**: This folder contains saved model checkpoints for the pretrained $f_\mathrm{aug}$. During training of TSAP, checkpoints for $f_\mathrm{det}$ and $\mathbb{a}$ are saved here as well.
- **/configs**: Configuration files for training TSAP.
- **/shell_scripts**: Scripts to train and/or test TSAP.
- **/src**: The source code for the core functionalities of TSAP.
- **main.py**: The main Python script for running TSAP.
