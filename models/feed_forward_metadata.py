import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNN(nn.Module):
    def __init__(self, input_dim=12, output_dim=18, n_hidden_1=256, n_hidden_2=128):
        super(FCNN, self).__init__()

        self.layer1 = nn.Linear(input_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.last = nn.Linear(n_hidden_2, output_dim)

        self.nonlin1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.Sotfmax = nn.Softmax()

    def forward(self, x):
        h1 = self.dropout(self.nonlin1(self.layer1(x)))
        h2 = self.dropout(self.nonlin1(self.layer2(h1)))
        output = self.Sotfmax(self.last(h2))
        return output
