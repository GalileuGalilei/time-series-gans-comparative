# load mitbih dataset

import os 
import numpy as np
import pandas as pd
import sys 
from tqdm import tqdm 
from scipy.stats import mode  # Para calcular a moda
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

cls_dit = {'Non-Ectopic Beats':0, 'Superventrical Ectopic':1, 'Ventricular Beats':2,
                                                'Unknown':3, 'Fusion Beats':4}

class load_and_resample_data(Dataset):
    def __init__(self, filename, label_column, seq_len, n_samples=10000):
        assert n_samples % seq_len == 0, "O número de linhas deve ser divisível por seq_len"

        data_train = pd.read_csv(filename)

        # apenas as colunas flowm flow packets/s e timestamp
        data_train = data_train[['Flow Packets/s', 'Flow Bytes/s', 'Timestamp', label_column]]

        # Converte 'Timestamp' para datetime com o formato específico
        data_train['Timestamp'] = pd.to_datetime(data_train['Timestamp'], errors='coerce')

        # Define 'Timestamp' como índice
        data_train.set_index('Timestamp', inplace=True)

        #sort pelo timestamp
        data_train.sort_index(inplace=True)

        # Convertendo as classes para números
        data_train[label_column] = pd.Categorical(data_train[label_column]).codes

        Y_train = data_train[label_column]
        X_train = data_train.drop([label_column], axis=1)

        # Drop de colunas não numéricas
        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        X_train = X_train.select_dtypes(include=[np.number])

        # resample usando a media do X_train e o ultimo label
        X_train = X_train.resample('20S').sum()
        Y_train = Y_train.resample('20S').last()

        #y_train with nan values, replace for 0
        Y_train.fillna(0, inplace=True)

        # remove os ultimos valores para fechar n_samples
        X_train = X_train.iloc[:n_samples]
        Y_train = Y_train.iloc[:n_samples]

        # Normalização usando MinMaxScaler
        scaler = MinMaxScaler()
        self.X_train = scaler.fit_transform(X_train)

        # Garantir que o número de amostras seja divisível por seq_len
        self.X_train = self.X_train[:len(self.X_train) // seq_len * seq_len]
        self.X_train = self.X_train.reshape(self.X_train.shape[0] // seq_len, seq_len, self.X_train.shape[1])
        self.X_train = np.transpose(self.X_train, (0, 2, 1))
        self.X_train = np.expand_dims(self.X_train, axis=2)  # (batch, seq_len, colunas, 1)

        # (batch, seq_len) - último valor da sequência
        self.Y_train = Y_train.values
        self.Y_train = self.Y_train.reshape(self.Y_train.shape[0] // seq_len, seq_len)[:, -1]

        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.Y_train.shape}')

        # Imprimir quantidades de amostras por classe
        for i in range(len(np.unique(self.Y_train))):
            print(f'Class {i} has {np.sum(self.Y_train == i)} samples')
        
    def __len__(self):
        return len(self.Y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]
