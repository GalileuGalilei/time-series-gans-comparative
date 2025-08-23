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
    features_to_train = ['Src Port', 'Dst Port', 'Bwd Init Win Bytes', 'Flow Packets/s', 'Fwd Packets/s', 'Bwd Packets/s', 'Flow IAT Mean', 'Bwd Header Length', 'Fwd Header Length', 'Flow Bytes/s']
    label_column = 'Stage'
    seq_len = 30 
    filename = "data/dapt2020.csv"
    
    #ja embaralhado por padrao
    data_set = DAPT2020(filename, label_column, seq_len, filter_features=features_to_train, is_train=True)
    data_set.shuffle()
    #data_set.order_by_class()  # Reordena os dados para que dados da mesma classe fiquem juntos nos dados de treino

    # ARGUMENTS
    opt = Options().parse()

    opt.seq_len = seq_len
    opt.data = "dapt2020"
    opt.iteration = 1000
    opt.hidden_dim = 150
    opt.num_layer = 3  
    opt.module = "gru" 
    opt.batch_size = 256
    opt.z_dim = 10
    opt.label_embed_dim = 10
    opt.num_classes = 5 # Number of unique classes in the training set

    # LOAD MODEL
    model = TimeGAN(opt, data_set.X_train, data_set.Y_train)

    # TRAIN MODEL
    model.train()

if __name__ == '__main__':
    train()
