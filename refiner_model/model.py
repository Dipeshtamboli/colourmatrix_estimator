import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, dilation=3, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 5, dilation=3, stride=2)
        self.conv3 = nn.Conv2d(16, 8, 5, dilation=3, stride=2)
        self.conv4 = nn.Conv2d(8, 3, 5, dilation=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1) 
	
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return self.avgpool(x)


