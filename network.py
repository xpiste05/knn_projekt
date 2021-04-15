import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv5 = nn.Conv2d(32, 64, 3)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(16000, 128)
        self.fc2 = nn.Linear(128, 36)
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

        x = x.view(-1, 16000)

        y1 = self.fc1(x)
        y1 = self.fc2(y1)
        y1 = self.softmax(y1)
        y2 = self.fc1(x)
        y2 = self.fc2(y2)
        y2 = self.softmax(y2)
        y3 = self.fc1(x)
        y3 = self.fc2(y3)
        y3 = self.softmax(y3)
        y4 = self.fc1(x)
        y4 = self.fc2(y4)
        y4 = self.softmax(y4)
        y5 = self.fc1(x)
        y5 = self.fc2(y5)
        y5 = self.softmax(y5)
        y6 = self.fc1(x)
        y6 = self.fc2(y6)
        y6 = self.softmax(y6)
        y7 = self.fc1(x)
        y7 = self.fc2(y7)
        y7 = self.softmax(y7)
        y8 = self.fc1(x)
        y8 = self.fc2(y8)
        y8 = self.softmax(y8)

        return y1, y2, y3, y4, y5, y6, y7, y8