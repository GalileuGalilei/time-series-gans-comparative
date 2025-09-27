# load mitbih dataset

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class DAPT2020(Dataset):
    def __init__(self, filename, label_column, seq_len, filter_features=None, train_test_split=0.7, is_train=True, attack_only=False, remove_outliers=True):
        self.is_train = is_train
        self.seq_len = seq_len
        self.attack_only = attack_only
        self.train_weighted_indices = None
        self.test_weighted_indices = None

        # Load data
        data_train = pd.read_csv(filename)

        # Filter columns
        if filter_features:
            filter_features = filter_features + ['Timestamp']
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
        self.Y_set = data_train[label_column].to_numpy()

        # Minmax scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.X_set = scaler.fit_transform(self.X_set)

        # Each sequence of size "seq_len" will predict the label of the last value in that sequence
        assert len(self.X_set) > seq_len, "The dataset is too small for the given sequence length."
        
        # Create sequences
        self.X_indices = np.array([np.arange(i,i + seq_len) for i in range(len(self.X_set) - seq_len - 1)])
        self.Y_indices = np.array([i + seq_len for i in range(len(self.Y_set) - seq_len - 1)])

        # Calculate train/test split
        self.train_size = int(len(self.X_indices) * train_test_split)
        self.train_indices = np.arange(0, self.train_size)
        self.test_indices = np.arange(self.train_size, len(self.X_indices))

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
        
    def shuffle(self, seed=42):
        '''
        Shuffle the dataset and re-split into train and test sets.
        '''

        # basic shuffle
        np.random.seed(seed)
        indices = np.arange(len(self.X_indices))
        np.random.shuffle(indices)

        self.train_indices = indices[:self.train_size]
        self.test_indices = indices[self.train_size:]
        
    def balance_classes(self, balance_test_set=False):
        '''
            Create weighted random samplers for balanced class sampling during training and testing.
        '''

        sample_weights = self.class_weights
        sample_weights = sample_weights[self.Y_set[self.train_indices]]

        train_weighted_sampler = WeightedRandomSampler(sample_weights, num_samples=len(self.train_indices), replacement=True)
        self.train_weighted_indices = list(train_weighted_sampler)

        if not balance_test_set:
            return

        sample_counts = np.bincount(self.Y_set[self.test_indices], minlength=len(self.classes_names))
        sample_weights = np.sqrt(1. / sample_counts)
        sample_weights = np.sqrt(sample_weights)
        sample_weights = sample_weights[self.Y_set[self.test_indices]]

        test_weighted_sampler = WeightedRandomSampler(sample_weights, num_samples=len(self.test_indices))
        self.test_weighted_indices = list(test_weighted_sampler)

    def expand(self): #todo: with indices do not work, fix later
        # expand dims to fit the TTS-CGAN input shape (channels, 1, seq_length)
        self.X_set = np.expand_dims(self.X_set, axis=1) # -> (batch, 1, channels)

    def one_hot_encode(self):
        self.Y_set = np.eye(len(self.classes_names))[self.Y_set]

    #nem ideia se isso funciona, provavelmente nao
    def order_by_class(self, batch_size): 
        # Order the dataset by class, so that each batch contains samples from only one class
        indices = np.arange(len(self.X_train_indices))
        ordered_indices = []
        for i in range(len(self.classes_names)):
            class_indices = indices[self.Y_set[self.X_train_indices] == i]
            np.random.shuffle(class_indices)
            ordered_indices.extend(class_indices[:batch_size])
        self.X_train_indices = self.X_train_indices[ordered_indices]
        self.Y_train_indices = self.Y_train_indices[ordered_indices]

    @property
    def X_test(self):
        if self.test_weighted_indices:
            return self.X_set[self.X_indices[self.test_indices[self.test_weighted_indices]]]
        else: 
            return self.X_set[self.X_indices[self.test_indices]]
    
    @property
    def Y_test(self):
        if self.test_weighted_indices:
            return self.Y_set[self.Y_indices[self.test_indices[self.test_weighted_indices]]]
        else:
            return self.Y_set[self.Y_indices[self.test_indices]]
    
    @property
    def X_train(self):
        if self.train_weighted_indices:
            return self.X_set[self.X_indices[self.train_indices[self.train_weighted_indices]]]
        else:
            return self.X_set[self.X_indices[self.train_indices]]

    @property
    def Y_train(self):
        if self.train_weighted_indices:
            return self.Y_set[self.Y_indices[self.train_indices[self.train_weighted_indices]]]
        else:
            return self.Y_set[self.Y_indices[self.train_indices]]
        
    @property
    def class_weights(self):
        sample_counts = np.bincount(self.Y_set[self.train_indices], minlength=len(self.classes_names))
        class_weights = np.sqrt(1. / sample_counts)
        #class_weights = np.sqrt(class_weights)  # Aplicar raiz quadrada
        return class_weights

    def __len__(self):
        if self.is_train:
            return len(self.train_indices)
        else:
            return len(self.test_indices)

    def __getitem__(self, idx):
        if self.is_train:
            if self.train_weighted_indices:
                idx = self.train_weighted_indices[idx]
            actual_idx = self.train_indices[idx]
        else:
            if self.test_weighted_indices:
                idx = self.test_weighted_indices[idx]
            actual_idx = self.test_indices[idx]

        x = self.X_set[self.X_indices[actual_idx]]
        y = self.Y_set[self.Y_indices[actual_idx]]
        return x, y