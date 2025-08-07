# load mitbih dataset

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class DAPT2020(Dataset):
    def __init__(self, filename, label_column, seq_len, filter_features=None, train_test_split=0.7, is_train=True, attack_only=False, remove_outliers=True):
        self.is_train = is_train
        self.seq_len = seq_len
        self.attack_only = attack_only

        # Load data
        data_train = pd.read_csv(filename)

        # Filter columns
        if filter_features:
            data_train = data_train[filter_features + [label_column]]

        # Sort by timestamp
        if 'Timestamp' in data_train.columns.to_list():
            data_train['Timestamp'] = pd.to_datetime(data_train['Timestamp'], errors='coerce')
            data_train.set_index('Timestamp', inplace=True)
            data_train.sort_index(inplace=True)
        else:
            print("Warning: 'Timestamp' column not found. Sorting by index instead.")

        # Lowercase
        data_train[label_column] = data_train[label_column].str.lower()

        # mapping labels to integers
        if attack_only:
            data_train[label_column] = data_train[label_column].map({
                'benign': 0,
                'reconnaissance': 1,
                'establish foothold': 1,
                'lateral movement': 1,
                'data exfiltration': 1
            })
            self.classes_names = ['benign', 'anomaly']  # Only two classes: benign and anomaly
        else:
            data_train[label_column] = data_train[label_column].map({
                'benign': 0,
                'reconnaissance': 1,
                'establish foothold': 2,
                'lateral movement': 3,
                'data exfiltration': 4
            })
            self.classes_names = ['benign', 'reconnaissance', 'establish foothold', 'lateral movement', 'data exfiltration']

        # To numeric
        data_train = data_train.apply(pd.to_numeric, errors='coerce')

        # Drop NaNs
        data_train = data_train.dropna(axis=1, how="any")
        self.features_names = data_train.columns.to_list()
        self.features_names.remove(label_column)

        # Remove outliers
        if remove_outliers:
            before_remove = len(data_train)
            for col in self.features_names:
                data_train[col] = stats.zscore(data_train[col])
                data_train = data_train[(np.abs(data_train[col]) < 4)]
            after_remove = len(data_train)

            print(f'{before_remove - after_remove} outliers removed')

        # Separate features and labels
        self.X_set = data_train.drop(columns=[label_column])
        self.Y_set = data_train[label_column]

        # Standard scaler is not recommended for time series preprocessing, review this in the future!
        #scaler = StandardScaler()
        #X_set = scaler.fit_transform(X_set)

        # Minmax scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.X_set = scaler.fit_transform(self.X_set)

        # Each sequence of size "seq_len" will predict the label of the last value in that sequence
        assert len(self.X_set) > seq_len, "The dataset is too small for the given sequence length."
        
        # Create sequences
        self.X_indices = np.array([np.arange(i,i + seq_len) for i in range(len(self.X_set) - seq_len - 1)])
        self.Y_indices = np.array([i + seq_len for i in range(len(self.Y_set) - seq_len - 1)])

        assert len(self.X_indices) == len(self.Y_indices), "X_set and Y_set must have the same length. Something is wrong."

        # Calculate train/test split
        self.train_size = int(len(self.X_indices) * train_test_split)

        # Print dataset info
        uniques = np.unique(self.Y_set)
        classes = len(uniques)
        i = 0
        print(f"Dataset loaded with {len(self.X_indices)} samples and {len(self.features_names)} features.")
        while i < classes:
            if attack_only and i == 0: # Se for apenas ataque, ignora a classe 0 (benigno)
                classes += 1
                i += 1
                continue
            total_per_class = len(self.Y_set[self.Y_set == i])
            print(f'Number of samples of class {self.classes_names[i]}: {total_per_class}')
            i += 1
        print(f"Train size: {self.train_size}, Test size: {len(self.X_indices) - self.train_size}")
        
    def shuffle(self, seed=22):
        np.random.seed(seed)
        
        indices = np.arange(len(self.Y_indices))
        np.random.shuffle(indices)

        self.X_indices = self.X_indices[indices]
        self.Y_indices = self.Y_indices[indices]

    def expand(self): #todo: with indices do not work, fix later
        # expand dims to fit the TTS-CGAN input shape (batch, channels, 1, seq_length)
        self.X_set = np.transpose(self.X_set, (0, 2, 1))
        self.X_set = np.expand_dims(self.X_set, axis=2) 

    def one_hot_encode(self):
        self.Y_set = np.eye(len(self.classes_names))[self.Y_set]

    @property
    def X_test(self):
        return self.X_set[self.X_indices[self.train_size:]]
    
    @property
    def Y_test(self):
        return self.Y_set[self.Y_indices[self.train_size:]]
    
    @property
    def X_train(self):
        return self.X_set[self.X_indices[:self.train_size]]
    
    @property
    def Y_train(self):
        return self.Y_set[self.Y_indices[:self.train_size]]

    def __len__(self):
        if self.is_train:
            return self.train_size
        else:
            return len(self.X_indices) - self.train_size
    
    def __getitem__(self, idx):
        if self.is_train:
            if idx >= self.train_size:
                raise IndexError("Index out of range for training set.")
            return self.X_set[self.X_indices[idx]], self.Y_set[self.Y_indices[idx]]
        else:
            return self.X_set[self.X_indices[idx + self.train_size]], self.Y_set[self.Y_indices[idx + self.train_size]]
