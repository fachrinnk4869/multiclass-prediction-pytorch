
import torch.nn as nn
import torch.nn.functional as F


class TestModel(nn.Module):
    def __init__(self, num_classes):
        super(TestModel, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # Adjust based on image size after conv layers
        # self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc1 = nn.Linear(3 * 128 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2)
        # x = F.relu(self.conv3(x))
        # x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
