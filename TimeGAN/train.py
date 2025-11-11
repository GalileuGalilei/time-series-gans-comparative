"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------

train.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from .options import Options
from .lib import TimeGAN
from data.data_loader import DAPT2020


def train():
    """ Training
    """


    # LOAD DATA
    seq_len = 128
    features_to_train = ['Src Port', 'Dst Port', 'Bwd Init Win Bytes', 'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s', 'Flow IAT Mean', 'Bwd Header Length', 'Fwd Header Length', 'Flow Bytes/s']
    train_set = DAPT2020("data/dapt2020.csv", "Stage", seq_len, filter_features=features_to_train, is_train=True, attack_only=False)
    train_set.shuffle()
    train_set.balance_classes()  # create balanced class indices for sampling

    #data_set.order_by_class()  # Reordena os dados para que dados da mesma classe fiquem juntos nos dados de treino

    # ARGUMENTS
    opt = Options().parse()

    opt.seq_len = seq_len
    opt.data = "dapt2020"
    opt.iteration = 5000
    opt.hidden_dim = 32
    opt.num_layer = 3  
    opt.module = "gru" 
    opt.batch_size = 64
    opt.z_dim = 10
    opt.label_embed_dim = 10
    opt.num_classes = 5 # Number of unique classes in the training set

    # LOAD MODEL
    model = TimeGAN(opt, train_set.X_train, train_set.Y_train)

    # TRAIN MODEL
    model.train()

if __name__ == '__main__':
    train()
