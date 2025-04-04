import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # x: [B, T, D]
        _, hidden = self.rnn(x)  # hidden: [num_layers, B, hidden_size]
        out = self.fc(hidden[-1])  # use last layer's hidden state
        return out


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # x: [B, T, D]
        output, (hidden, cell) = self.lstm(x)  # hidden: [num_layers, B, hidden_size]
        out = self.fc(hidden[-1])  # Use last layer's hidden state
        return out
