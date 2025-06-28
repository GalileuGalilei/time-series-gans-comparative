import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hmmlearn import hmm

class IClassifier:
    def is_torch_model(self):
        """
        Check if the classifier is a PyTorch model.
        """
        return isinstance(self, nn.Module)
    
    def get_name(self):
        """
        Get the name of the classifier.
        """
        return self.__class__.__name__
    
    def copy(self):
        """
        Create a copy of the classifier.
        """
        return self.__class__()
    

class RandomForestClassifierModel(IClassifier):
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
    
    def get_name(self):
        return "RandomForestClassifierModel"

    def copy(self):
        return RandomForestClassifierModel(self.classifier.n_estimators, self.classifier.max_depth)
    

class SVMClassifier(IClassifier):
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
    
    def get_name(self):
        return "SVMClassifier"

    def copy(self):
        return SVMClassifier(self.classifier.kernel, self.classifier.C)


class LSTMClassifier(nn.Module, IClassifier):
    def __init__(self, n_channels, seq_length, hidden_dim=64, n_classes=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=n_channels, hidden_size=hidden_dim, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, n_classes)

        self.n_channels = n_channels
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

    def forward(self, x):
        # x: (batch_size, n_channels, 1, seq_length)
        x = x.squeeze(2)  # → (batch_size, n_channels, seq_length)
        x = x.permute(0, 2, 1)  # → (batch_size, seq_length, n_channels)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output  # Softmax é aplicado fora, se necessário

    def copy(self):
        return LSTMClassifier(self.n_channels, self.seq_length, self.hidden_dim, self.n_classes)

    def get_name(self):
        return "LSTMClassifier"

class TransformerClassifier(nn.Module, IClassifier):
    def __init__(self, n_channels, seq_length, n_classes=2):
        super(TransformerClassifier, self).__init__()
        self.n_channels = n_channels
        self.seq_length = seq_length
        self.n_classes = n_classes
        self.transformer = nn.TransformerEncoderLayer(d_model=n_channels, nhead=3, dim_feedforward=64, dropout=0.3)
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
    
    def get_name(self):
        return "TransformerClassifier"

    