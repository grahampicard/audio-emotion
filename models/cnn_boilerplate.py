import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 64, 3)

        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(4, stride=4)

        self.fc1 = nn.Linear(12480, 256)
        self.fc2 = nn.Linear(256, 18)

        self.drp = nn.Dropout2d(0.25)


    def forward(self, x):
        x = self.drp(self.pool1(F.relu(self.conv1(x))))
        x = self.drp(self.pool1(F.relu(self.conv2(x))))
        x = self.drp(self.pool1(F.relu(self.conv3(x))))
        x = self.drp(self.pool1(F.relu(self.conv4(x))))
        x = self.drp(self.pool2(F.relu(self.conv5(x))))
        x = x.view(-1, 12480)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.conv4 = nn.Conv2d(16, 8, 3)
        self.conv5 = nn.Conv2d(8, 2, 3)

        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pool2 = nn.MaxPool2d(4, stride=4)

        self.fc1 = nn.Linear(390, 36)
        self.fc2 = nn.Linear(36, 18)

        self.drp = nn.Dropout2d(0.25)


    def forward(self, x):
        x = self.drp(self.pool1(F.relu(self.conv1(x))))
        x = self.drp(self.pool1(F.relu(self.conv2(x))))
        x = self.drp(self.pool1(F.relu(self.conv3(x))))
        x = self.drp(self.pool1(F.relu(self.conv4(x))))
        x = self.drp(self.pool2(F.relu(self.conv5(x))))
        x = x.view(-1, 390)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)
