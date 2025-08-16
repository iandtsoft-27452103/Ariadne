import torch
import torch.nn as nn
import torch.nn.functional as F

class gru(nn.Module):
    def __init__(self):
        super(gru, self).__init__()
        self.batch_size = 0
        self.emb = nn.Embedding(4096, 64)
        self.l1 = nn.GRU(input_size=64, hidden_size=64, num_layers=4, batch_first=True, bidirectional=True)
        #self.l1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=4, batch_first=True, bidirectional=True)
        self.l2 = nn.Linear(128*128, 5864)

    def forward(self, x):
        h = self.emb(x)
        h = self.l1(h)
        h = h[0].reshape(self.batch_size, 128*128)
        h = self.l2(h)
        return h