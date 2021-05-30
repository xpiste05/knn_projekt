import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LocalizationNetwork(nn.Module):
    def __init__(self, numOfControlPoints=10):
        super().__init__()

        self.numOfControlPoints = numOfControlPoints

        self.pool = nn.MaxPool2d(2, 2)
        self.aPool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(512, 256)
        
        self.fc2 = nn.Linear(256, numOfControlPoints * 2)

        self.init_stn()

    def forward(self, x):

        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.pool(x)
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.aPool(x)

        x = x.view(x.size()[0], -1)

        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)

        x = x.view(-1, 2, self.numOfControlPoints)
        
        return x

    def init_stn(self):

        interval = np.linspace(0.05, 0.95, self.numOfControlPoints // 2)
        controlPoints = [[],[]]

        for y in [0.1,0.9]:
            for i in range(self.numOfControlPoints // 2):
                controlPoints[1].append(y)
            for x in interval:
                controlPoints[0].append(x)
        
        self.fc2.weight.data.zero_()
        self.fc2.bias.data = torch.Tensor(controlPoints).view(-1).float().to(device)
