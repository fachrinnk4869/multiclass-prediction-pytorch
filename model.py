import torch.nn as nn
import torch.nn.functional as F


class TestModel(nn.Module):
    def __init__(self, num_classes):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # Calculate the output size after the convolutional and pooling layers (128 * 16 * 16)
        # self.fc1 = nn.Linear(128 * 128 * 3, 256)  # ->> dnn
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # ->> cnn
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 128x128 -> 64x64
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 64x64 -> 32x32
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)  # 32x32 -> 16x16
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
