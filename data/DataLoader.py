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

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class load_and_preprocess_data(Dataset):
    def __init__(self, filename, features_names, label_column, seq_len, attack_only=False, is_train=True, shuffle=True, seed=22, expand=True, one_hot=False):
        # Carregar dataset
        self.is_train = is_train
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
        data_train[label_column] = data_train[label_column].astype('category')  
        self.classes = dict(enumerate(data_train[label_column].cat.categories)) 

        # Converter labels para valores numéricos
        data_train[label_column] = data_train[label_column].cat.codes

        # Converte as todas as colunas para numerico
        data_train = data_train.apply(pd.to_numeric, errors='coerce')

        #remove timestamp de features_names
        features_names.remove('Timestamp')

        # Remove outliers
        before_remove = len(data_train)
        for col in features_names:
            data_train[col] = data_train[col].fillna(data_train[col].mean())
            data_train[col] = stats.zscore(data_train[col])
            data_train = data_train[(np.abs(data_train[col]) < 3)]
        after_remove = len(data_train)

        print(f'Foram removidos {before_remove - after_remove} outliers')


        # Resample: média para X_train, última observação para Y_train
        data_resampled = data_train#.resample('0.5S').agg({**{col: 'mean' for col in features_names}, label_column: 'last'})

        # Remover NaNs
        data_resampled.dropna(inplace=True)

        # Separar features e labels
        X_set = data_resampled.drop(columns=[label_column])
        Y_set = data_resampled[label_column]

        # Remove os últimos valores para ficar um número divisível por seq_len.
        size = len(Y_set)
        #acha o maior múltiplo de seq_len menor que size
        seq_len = 30
        while size % seq_len != 0:
            size += 1

        #repete o último valor de Y_set e X_set para completar o tamanho
        if not Y_set.empty:
            Y_set = pd.concat([Y_set, pd.Series(Y_set.iloc[-1]).repeat(size - len(Y_set))]).reset_index(drop=True)
        if not X_set.empty:
            X_set = pd.concat([X_set, pd.DataFrame([X_set.iloc[-1]] * (size - len(X_set)), columns=X_set.columns)]).reset_index(drop=True)

        # Normalização usando entre -1 e 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.X_set = scaler.fit_transform(X_set)

        # Garantir que o número de amostras seja divisível por seq_len
        self.X_set = self.X_set[:len(self.X_set) // seq_len * seq_len]
        self.X_set = self.X_set.reshape(-1, seq_len, self.X_set.shape[1])
        if expand:
            self.X_set = np.transpose(self.X_set, (0, 2, 1))
            self.X_set = np.expand_dims(self.X_set, axis=2) 

        # Ordena as classes por frequência (menor para maior)
        rarity_order = list(np.argsort(np.bincount(Y_set)))
        rarity_rank = {cls: i for i, cls in enumerate(rarity_order)}

        # Inicializa Y_set com zeros no formato reduzido
        self.Y_set = np.zeros(len(Y_set) // seq_len, dtype=int)

        # Agrupa classes priorizando as menos comuns
        for i in range(len(self.Y_set)):
            agg_class = rarity_order[-1]  # começa com a classe mais comum
            for j in range(seq_len):
                current_class = Y_set[i * seq_len + j]
                if rarity_rank[current_class] < rarity_rank[agg_class]:
                    agg_class = current_class
            self.Y_set[i] = agg_class


        if shuffle:
            np.random.seed(seed)
            # Embaralhar os dados
            indices = np.arange(len(self.Y_set))
            np.random.shuffle(indices)
            self.X_set = self.X_set[indices]
            self.Y_set = self.Y_set[indices]

        # Dividir os dados em treino e teste
        cutoff = int(len(self.Y_set) * 0.7)
        
        self.X_train_set = self.X_set[:cutoff]
        self.Y_train_set = self.Y_set[:cutoff]
        
        self.X_test_set = self.X_set[cutoff:]
        self.Y_test_set = self.Y_set[cutoff:]

        if attack_only:
            # Se for apenas ataque, remove benignos
            self.X_set = self.X_set[self.Y_set != 0]
            self.Y_set = self.Y_set[self.Y_set != 0]

        print(f'X shape is {self.X_set.shape}')
        print(f'Y shape is {self.Y_set.shape}')

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
            # One-hot encoding
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
