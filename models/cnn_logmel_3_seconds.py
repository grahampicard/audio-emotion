""" CNN models 

    Input:
        - 3 second samples
        - 32k sample rate

    Output:
        - 18 emotional tags as output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_simple_3s_32k(nn.Module):
    """ Basic model architecture taken from a github page """

    def __init__(self):
        super(CNN_simple_3s_32k, self).__init__()

        self.conv1 = nn.Conv2d(1, 128, 2)
        self.conv5 = nn.Conv2d(128, 64, 2)

        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(4, stride=2)

        self.fc1 = nn.Linear(86400, 256)
        self.fc2 = nn.Linear(256, 18)

        self.drp = nn.Dropout2d(0.25)


    def forward(self, x):
        x = self.drp(self.pool1(F.relu(self.conv1(x))))
        x = self.drp(self.pool2(F.relu(self.conv5(x))))
        x = x.view(-1, 86400)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # return F.softmax(x, dim=1)
        return F.sigmoid(x)