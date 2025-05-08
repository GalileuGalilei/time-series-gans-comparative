#!/usr/bin/env python
#https://github.com/MadaneA/MTS-CGAN

import os
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default="0")
    parser.add_argument('--node', type=str, default="0015")
    return parser.parse_args()

args = parse_args()

# Definir variáveis de ambiente apenas para o subprocesso
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
#debug
env["CUDA_LAUNCH_BLOCKING"] = "1"
env["TORCH_USE_CUDA_DSA"] = "1"


# Construir a lista de argumentos
command = [
    "python", "trainTTSCGAN.py",
    "-gen_bs", "32",
    "-dis_bs", "32",
    "--dist-url", "tcp://localhost:4321",
    "--dist-backend", "nccl",
    "--world-size", "1",
    "--max_epoch", "90",
    "--rank", args.rank,
  #  "--gpu", "0",
    "--dataset", "mitbith",
    "--bottom_width", "8",
    "--max_iter", "500000",
    "--img_size", "32",
    "--gen_model", "my_gen",
    "--dis_model", "my_dis",
    "--df_dim", "384",
    "--d_heads", "4",
    "--d_depth", "3",
    "--g_depth", "5,4,2",
   # "--load_path", "logs/TTS_APT_CGAN_2025_04_15_12_41_54/Model/checkpoint",
    "--dropout", "0",
    "--latent_dim", "100",
    "--gf_dim", "1024",
    "--num_workers", "8",
    "--g_lr", "0.0001",
    "--d_lr", "0.0003",
    "--optimizer", "adam",
    "--loss", "lsgan",
    "--wd", "1e-3",
    "--beta1", "0.9",
    "--beta2", "0.999",
    "--phi", "1",
    "--batch_size", "30",
    "--num_eval_imgs", "50000",
    "--init_type", "xavier_uniform",
    "--n_critic", "1",
    "--val_freq", "20",
    "--print_freq", "20",
    "--grow_steps", "0", "0",
    "--fade_in", "0",
    "--patch_size", "8",
    "--ema_kimg", "500",
    "--ema_warmup", "0.1",
    "--ema", "0.9999",
    "--diff_aug", "translation,cutout,color",
    "--exp_name", "TTS_PURE_ORIGINAL_APT_CGAN"
]

# Executar o comando
subprocess.run(command, env=env, check=True)
