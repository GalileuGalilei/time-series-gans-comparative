# load mitbih dataset

import os 
import numpy as np
import pandas as pd
import sys 
from scipy import stats
from tqdm import tqdm 
from scipy.stats import mode  # Para calcular a moda
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class load_and_preprocess_data(Dataset):
    def order_by_class(self):
        """
        Reordena os dados para que dados da mesma classe fiquem juntos nos dados de treino.
        """

        if not self.one_hot:
            #pega os indices de Y_train_set ordenados por classe
            indices = np.argsort(self.Y_train_set)
            # Reordena X_train_set e Y_train_set de acordo com os indices
            self.X_train_set = self.X_train_set[indices]
            self.Y_train_set = self.Y_train_set[indices]
        else:
            # Para one-hot encoding, precisamos ordenar de forma diferente
            # Pega os indices de Y_train_set ordenados por classe
            indices = np.argsort(np.argmax(self.Y_train_set, axis=1))
            # Reordena X_train_set e Y_train_set de acordo com os indices
            self.X_train_set = self.X_train_set[indices]
            self.Y_train_set = self.Y_train_set[indices]


    def __init__(self, filename, features_names, label_column, seq_len, attack_only=False, is_train=True, shuffle=True, seed=22, expand=True, one_hot=False):
        # Carregar dataset
        self.is_train = is_train
        self.one_hot = one_hot
        data_train = pd.read_csv(filename)

        # Selecionar apenas as colunas relevantes
        data_train = data_train[features_names + [label_column]]

        # Converter 'Timestamp' para datetime e definir como índice
        data_train['Timestamp'] = pd.to_datetime(data_train['Timestamp'], errors='coerce')
        data_train.set_index('Timestamp', inplace=True)
        data_train.sort_index(inplace=True)

        # Converter a label para lowercase
        data_train[label_column] = data_train[label_column].str.lower()

        # Criar um mapeamento label -> número
        data_train[label_column] = data_train[label_column].map({
            'benign': 0,
            'reconnaissance': 1,
            'establish foothold': 2,
            'lateral movement': 3,
            'data exfiltration': 4
        })
        # Fill any unmapped or missing values with a default (e.g., 0 or -1)
        #data_train[label_column] = data_train[label_column].fillna(0).astype(int)

        self.classes = ['benign', 'reconnaissance', 'establish foothold', 'lateral movement', 'data exfiltration']

        # Converte as todas as colunas para numerico
        data_train = data_train.apply(pd.to_numeric, errors='coerce')

        #remove timestamp de features_names
        features_names.remove('Timestamp')
        self.features_names = features_names.copy()

        # Remove outliers
        before_remove = len(data_train)
        for col in features_names:
            data_train[col] = data_train[col].fillna(data_train[col].mean())
            data_train[col] = stats.zscore(data_train[col])
            data_train = data_train[(np.abs(data_train[col]) < 3)]
        after_remove = len(data_train)

        print(f'{before_remove - after_remove} outliers removed')

        # Remove NaNs
        data_train.dropna(inplace=True)

        # Separate features and labels
        X_set = data_train.drop(columns=[label_column])
        Y_set = data_train[label_column]

        # Standard scaler
        scaler = StandardScaler()
        X_set = scaler.fit_transform(X_set)

        # Minmax scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_set = scaler.fit_transform(X_set)

        # Each sequence of size "seq_len" will predict the label of the last value in that sequence
        assert len(X_set) > seq_len, "The dataset is too small for the given sequence length."
        
        # Create sequences
        self.X_set = np.array([X_set[i:i + seq_len] for i in range(len(X_set) - seq_len - 1)])
        self.Y_set = np.array([Y_set[i + seq_len] for i in range(len(Y_set) - seq_len - 1)])

        assert len(self.X_set) == len(self.Y_set), "X_set and Y_set must have the same length."

        # expand dims to fit the TTS-CGAN input shape (batch, channels, 1, seq_length)
        if expand:
            self.X_set = np.transpose(self.X_set, (0, 2, 1))
            self.X_set = np.expand_dims(self.X_set, axis=2) 

        if shuffle:
            np.random.seed(seed)
            # Embaralhar os dados
            indices = np.arange(len(self.Y_set))
            np.random.shuffle(indices)
            self.X_set = self.X_set[indices]
            self.Y_set = self.Y_set[indices]

        # Train and test split
        cutoff = int(len(self.Y_set) * 0.7)
        
        self.X_train_set = self.X_set[:cutoff]
        self.Y_train_set = self.Y_set[:cutoff]
        
        self.X_test_set = self.X_set[cutoff:]
        self.Y_test_set = self.Y_set[cutoff:]

        print(f'X train shape is {self.X_train_set.shape}')
        print(f'Y train shape is {self.Y_train_set.shape}')

        # Imprimir quantidades de amostras por classe
        uniques = np.unique(self.Y_set)
        classes = len(uniques)
        i = 0
        while i < classes:
            if attack_only and i == 0: # Se for apenas ataque, ignora a classe 0 (benigno)
                classes += 1
                i += 1
                continue
            total_per_class = len(self.Y_set[self.Y_set == i])
            print(f'Quantidade de amostras da classe {self.classes[i]}: {total_per_class}')
            i += 1

        if one_hot:
            self.Y_set = np.eye(len(self.classes))[self.Y_set]
            self.Y_train_set = np.eye(len(self.classes))[self.Y_train_set]
            self.Y_test_set = np.eye(len(self.classes))[self.Y_test_set]
        
    def __len__(self):
        if self.is_train:
            return len(self.X_train_set)
        else:
            return len(self.X_test_set)
    
    def __getitem__(self, idx):
        if self.is_train:
            return self.X_train_set[idx], self.Y_train_set[idx]
        else:
            return self.X_test_set[idx], self.Y_test_set[idx]
