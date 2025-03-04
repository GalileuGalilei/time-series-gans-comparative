# load mitbih dataset

import os 
import numpy as np
import pandas as pd
import sys 
from tqdm import tqdm 
from scipy.stats import mode  # Para calcular a moda
from sklearn.preprocessing import StandardScaler

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
    def __init__(self, filename, label_column, seq_len, n_samples=20000):

        assert n_samples % seq_len == 0, "O número de linhas deve ser divisível por seq_len"

        data_train = pd.read_csv(filename)
        data_train = resample(data_train, n_samples=n_samples, random_state=123, replace=True)

        # Convertendo as classes para números
        labels_names = data_train[label_column].unique()
        data_train[label_column].replace(labels_names, [i for i in range(len(labels_names))], inplace=True)

        Y_train = data_train[label_column]
        X_train = data_train.drop([label_column], axis=1)

        #temporario
        keep_columns_with_names = ['Flow Bytes/s']
        X_train = X_train[keep_columns_with_names]


        #drop non-numeric columns
        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        X_train = X_train.select_dtypes(include=[np.number])
        
        #standardize, testes ainda
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)

        # (batch, 1, colunas, seq_len)
        #self.X_train = X_train.values
        self.X_train = self.X_train.reshape(self.X_train.shape[0] // seq_len, seq_len, self.X_train.shape[1])
        self.X_train = np.transpose(self.X_train, (0, 2, 1))
        self.X_train = np.expand_dims(self.X_train, axis=2) 

        # (batch, seq_len)
        self.Y_train = Y_train.values
        self.Y_train = self.Y_train.reshape(self.Y_train.shape[0] // seq_len, seq_len)[:,-1]
            
        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.Y_train.shape}')

        #print how mush of each class we have
        for i in range(len(np.unique(self.Y_train))):
            print(f'Class {i} has {np.sum(self.Y_train == i)} samples')
        
    def __len__(self):
        return len(self.Y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]