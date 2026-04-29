import torch
import torch.nn as nn


class SimpleLSTMNetwork(nn.Module):

    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.forward_layer = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        h0 = torch.zeros((1, batch_size, hidden_dim)).to(device)
        c0 = torch.zeros((1, batch_size, hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden

    def forward(self, x):
        embedded = self.embedding(x)
        h0 = self.init_hidden(x.shape[0])
        output, (hidden, cell) = self.lstm_layer(embedded, h0)
        hidden = hidden[-1, :, :]
        logits = torch.sigmoid(self.forward_layer(hidden))
        return logits


class BiLSTMModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super().__init__()
        num_classes = 1
        self.embedding  = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.forward_layer = nn.Linear(hidden_dim * 2, num_classes)

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        h0 = torch.zeros((2, batch_size, hidden_dim)).to(device)
        c0 = torch.zeros((2, batch_size, hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden

    def forward(self, x):
        embedded = self.embedding(x)

        h0 = self.init_hidden(x.shape[0])
        output, (hidden, cell) = self.lstm_layer(embedded, h0)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        logits = torch.sigmoid(self.forward_layer(hidden))
        return logits

