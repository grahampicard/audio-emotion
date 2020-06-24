import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNN(nn.Module):
    def __init__(self, input_dim=12, output_dim=18, n_hidden_1=72, n_hidden_2=36, num_classes=18):
        super(FCNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer1 = nn.Linear(input_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, num_classes)

        self.nonlin1 = nn.Sigmoid()
        self.nonlin2 = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h1 = self.dropout(self.nonlin1(self.layer1(x)))
        h2 = self.dropout(self.nonlin2(self.layer2(h1)))
        output = self.softmax(self.layer3(h2))

        return output