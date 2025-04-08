import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, n_channels, seq_length, hidden_dim=64, n_classes=2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=seq_length, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x):
        x = x.squeeze(2)  
        lstm_out, _ = self.lstm(x)  
        last_hidden = lstm_out[:, -1, :] 
        output = self.fc(last_hidden)  
        return self.softmax(output)  


class CNNCassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 64, 6)
        self.conv2 = nn.Conv1d(64, 64, 6)
        self.conv3 = nn.Conv1d(64, 64, 3)
        self.dropout = nn.Dropout(p=0.5) 
        self.pool = nn.MaxPool1d(3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(384, 100)
        self.fc2 = nn.Linear(100, 5)

    def forward(self, x):
        x = x.squeeze(2)  
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

    