from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from skorch import NeuralNetClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        X = X.reshape(X.shape[0], -1)  # Reshape X to (batch, 30*channels)
        self.classifier.fit(X, y)

    def predict(self, X):
        X = np.array(X)  # Ensure X is a NumPy array
        X = X.reshape(X.shape[0], -1)  # Reshape X to (batch, 30*channels)
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
        X = X.reshape(X.shape[0], -1)  # Reshape X to (batch, 30*channels)
        self.classifier.fit(X, y)

    def predict(self, X):
        X = np.array(X)  # Ensure X is a NumPy array
        X = X.reshape(X.shape[0], -1)  # Reshape X to (batch, 30*channels)
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
        if x.ndim == 2:
            x = x.view(-1, self.seq_length, self.n_channels)
            # x shape: (batch, seq_length, n_channels)

        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output  # Softmax é aplicado fora, se necessário

    def copy(self):
        return LSTMClassifier(self.n_channels, self.seq_length, self.hidden_dim, self.n_classes)

    def get_name(self):
        return "LSTMClassifier"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:(d_model // 2)])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerClassifier(nn.Module, IClassifier):
    def __init__(self, n_channels, seq_length, n_classes=2):
        super(TransformerClassifier, self).__init__()
        self.n_channels = n_channels
        self.seq_length = seq_length
        self.n_classes = n_classes
        self.pos_encoder = PositionalEncoding(d_model=n_channels, max_len=seq_length)
        #todo: colocar o nhead como parâmetro
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_channels, nhead=2, dim_feedforward=32, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(n_channels * seq_length, n_classes)

    def forward(self, x):
        #x = (batch, seq_length, n_channels)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (seq_length, batch, n_channels) for transformer
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (batch, seq_length, n_channels)
        x = x.contiguous().view(x.size(0), -1)  # Flatten for the fully connected layer
        output = self.fc(x)  # Final classification layer
        return output
    
    def copy(self):
        return TransformerClassifier(self.n_channels, self.seq_length, self.n_classes)
    
    def get_name(self):
        return "TransformerClassifier"