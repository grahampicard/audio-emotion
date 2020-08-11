""" CNN models 

    Input:
        - 30 second samples
        - 32k sample rate

    Output:
        - 18 emotional tags as output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_simple_30s_32k(nn.Module):
    """ Basic model architecture taken from a github page """

    def __init__(self):
        super(CNN_simple_30s_32k, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 64, 3)

        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(4, stride=4)

        self.fc1 = nn.Linear(26880, 256)
        self.fc2 = nn.Linear(256, 18)

        self.drp = nn.Dropout2d(0.25)


    def forward(self, x):
        x = self.drp(self.pool1(F.relu(self.conv1(x))))
        x = self.drp(self.pool1(F.relu(self.conv2(x))))
        x = self.drp(self.pool1(F.relu(self.conv3(x))))
        x = self.drp(self.pool1(F.relu(self.conv4(x))))
        x = self.drp(self.pool2(F.relu(self.conv5(x))))
        x = x.view(-1, 26880)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class CNN_dev_30s_32k(nn.Module):
    """ Simplified model to enable more rapid development"""

    def __init__(self):
        super(CNN_dev_30s_32k, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.conv4 = nn.Conv2d(16, 8, 3)
        self.conv5 = nn.Conv2d(8, 2, 3)

        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(4, stride=4)

        self.fc1 = nn.Linear(840, 256)
        self.fc2 = nn.Linear(256, 18)

        self.drp = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.drp(self.pool1(F.relu(self.conv1(x))))
        x = self.drp(self.pool1(F.relu(self.conv2(x))))
        x = self.drp(self.pool1(F.relu(self.conv3(x))))
        x = self.drp(self.pool1(F.relu(self.conv4(x))))
        x = self.drp(self.pool2(F.relu(self.conv5(x))))
        x = x.view(-1, 840)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
