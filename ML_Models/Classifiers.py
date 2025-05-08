import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hmmlearn import hmm

class LSTMClassifier(nn.Module):
    def __init__(self, n_channels, seq_length, hidden_dim=64, n_classes=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=seq_length, hidden_size=hidden_dim, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x):
        x = x.squeeze(2)  
        lstm_out, _ = self.lstm(x)  
        last_hidden = lstm_out[:, -1, :] 
        output = self.fc(last_hidden)  
        return self.softmax(output)  

class RandomForestClassifierModel:
    def __init__(self, n_estimators=100, max_depth=None):
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def fit(self, X, y):
        X = np.array(X)  # Ensure X is a NumPy array
        y = np.array(y)  # Ensure y is a NumPy array
        X = X.squeeze(2).reshape(X.shape[0], -1)  # Reshape X to (batch, 30*channels)
        self.classifier.fit(X, y)

    def predict(self, X):
        X = np.array(X)  # Ensure X is a NumPy array
        X = X.squeeze(2).reshape(X.shape[0], -1)  # Reshape X to (batch, 30*channels)
        return self.classifier.predict(X)
    
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.classifier, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.classifier = pickle.load(f)

    def copy(self):
        return RandomForestClassifierModel(self.classifier.n_estimators, self.classifier.max_depth)
    

class SVMClassifier:
    def __init__(self, kernel='rbf', C=1.0):
        self.classifier = SVC(kernel=kernel, C=C)

    def fit(self, X, y):
        X = np.array(X)  # Ensure X is a NumPy array
        y = np.array(y)  # Ensure y is a NumPy array
        X = X.squeeze(2).reshape(X.shape[0], -1)  # Reshape X to (batch, 30*channels)
        self.classifier.fit(X, y)

    def predict(self, X):
        X = np.array(X)  # Ensure X is a NumPy array
        X = X.squeeze(2).reshape(X.shape[0], -1)  # Reshape X to (batch, 30*channels)
        return self.classifier.predict(X)
    
    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.classifier, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.classifier = pickle.load(f)


class TransformerClassifier(nn.Module):
    def __init__(self, n_channels, seq_length, n_classes=2):
        super(TransformerClassifier, self).__init__()
        self.n_channels = n_channels
        self.seq_length = seq_length
        self.n_classes = n_classes
        self.transformer = nn.TransformerEncoderLayer(d_model=n_channels, nhead=4, dim_feedforward=64, dropout=0.3)
        self.fc = nn.Linear(n_channels * seq_length, n_classes)

    def forward(self, x):
        x = x.squeeze(2)  # Remove the singleton dimension
        x = x.permute(0, 2, 1)  # Change shape to (batch, channels, seq_length)
        x = self.transformer(x)  # Apply the transformer
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        output = self.fc(x)  # Final classification layer
        return output
    
    def copy(self):
        return TransformerClassifier(self.n_channels, self.seq_length, self.n_classes)

    