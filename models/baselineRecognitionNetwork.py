import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineRecognitionNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc11 = nn.Linear(8000, 128)
        self.fc21 = nn.Linear(128, 36)

        self.fc12 = nn.Linear(8000, 128)
        self.fc22 = nn.Linear(128, 36)

        self.fc13 = nn.Linear(8000, 128)
        self.fc23 = nn.Linear(128, 36)

        self.fc14 = nn.Linear(8000, 128)
        self.fc24 = nn.Linear(128, 36)

        self.fc15 = nn.Linear(8000, 128)
        self.fc25 = nn.Linear(128, 36)

        self.fc16 = nn.Linear(8000, 128)
        self.fc26 = nn.Linear(128, 36)

        self.fc17 = nn.Linear(8000, 128)
        self.fc27 = nn.Linear(128, 36)

        self.fc18 = nn.Linear(8000, 128)
        self.fc28 = nn.Linear(128, 36)

        self.softmax = nn.Softmax()
        

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.pool(x)

        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.pool(x)

        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.pool(x)

        x = x.view(-1, 8000)

        y1 = self.fc11(x)
        y1 = self.fc21(y1)

        y2 = self.fc12(x)
        y2 = self.fc22(y2)

        y3 = self.fc13(x)
        y3 = self.fc23(y3)

        y4 = self.fc14(x)
        y4 = self.fc24(y4)

        y5 = self.fc15(x)
        y5 = self.fc25(y5)

        y6 = self.fc16(x)
        y6 = self.fc26(y6)

        y7 = self.fc17(x)
        y7 = self.fc27(y7)

        y8 = self.fc18(x)
        y8 = self.fc28(y8)

        return y1, y2, y3, y4, y5, y6, y7, y8
