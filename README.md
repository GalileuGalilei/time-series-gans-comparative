# 🧠 Synthetic APT Generation Using Time Series-based GANs

This repository contains the source code, configuration files, and evaluation scripts for the experiments presented in the paper:

> **Comparative Study of GAN-Based Synthetic Data
Generation for APT Detection in Cybersecurity**
> Authors: Alfredo Cossetin Neto, Carlos Raniery, Luis Alvaro, Vinicius Garcia
> Submitted to: soon  
> Year: soon

## 📌 Overview

This work investigates the generation of synthetic network traffic sequences representing Advanced Persistent Threat (APT) stages using three different GAN-based models:  

- **TTS-CGAN**: Transformer-based, class-conditional GAN  
- **RCGAN**: RNN-based conditional GAN  
- **TimeGAN**: Hybrid model with autoencoder, supervisor, and adversarial loss

The models are trained on the [DAPT2020](https://doi.org/10.1109/DAPT2020Dataset) dataset to generate synthetic sequences representing multiple stages of APTs.  

The generated data is evaluated using:
- **PCA/t-SNE projections** (for qualitative analysis)
- **DTW distance** (for statistical similarity)
- **ML classification performance** on synthetic and semi-synthetic datasets

## 📁 Repository Structure
todo


## 🚀 Getting Started

### 1. Requirements
todo

### 2. Download DAPT2020
- download the csv files at: https://www.kaggle.com/datasets/sowmyamyneni/dapt2020
- place them under `data/`

### 3. Training
- python -m TTSCGAN.train
- python -m TimeGAN.train
- python -m RGAN.train

### Evaluation
- `data_utils.py` has all the necessary code for data generation
- `metrics.py` and `TSTR.py` scripts have all the experiments and evaluation metrics

  📊 Results
## Quantitative results (DTW scores and classification accuracy) and qualitative PCA plots can be found in the figures/ directory or reproduced via the evaluation scripts.

@article{soon enough, hopefully}
