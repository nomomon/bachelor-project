import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, num_classes, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()

        # LSTM layers process the vector sequences
        self.lstm_1 = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.lstm_2 = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True)
        
        # Dense layer to predict 
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x is of shape [B, T, E]

        x = x.permute(1, 0, 2)
        # x is of shape [T, B, E]
        
        # LSTM layers
        output, _ = self.lstm_1(x)
        _, (hidden, _) = self.lstm_2(output)

        # hidden is of shape [2, B, H]

        # Concat the final 
        #    forward  (hidden[-2,:,:]) and 
        #    backward (hidden[-1,:,:])
        # hidden layers and  apply dropout

        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = F.dropout(hidden, p=0.5, training=self.training)

        # hidden is of shape [B, H * 2]

        # FC
        logits = self.fc(hidden)

        return logits