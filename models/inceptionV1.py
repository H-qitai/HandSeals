import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionV1(nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionV1, self).__init__()
        # Initial convolution layers to process the input image
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)  # First conv layer with 32 filters
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)  # Second conv layer, keeps the channel size
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Third conv layer, increases channels to 64, with padding
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # Pooling to reduce the spatial size

        # Two Inception modules to extract complex features from the data
        self.inception1 = InceptionModule(64, 32, 64, 32, 64, 32)  # First inception module
        self.inception2 = InceptionModule(160, 64, 128, 64, 128, 64)  # Second inception module

        # Adaptive average pooling to reduce each feature map to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)  # Dropout for regularization
        self.fc = nn.Linear(320, num_classes)  # Final fully connected layer

    def forward(self, x):
        # Forward pass through initial convolution layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool1(x)

        # Forward pass through inception modules
        x = self.inception1(x)
        x = self.inception2(x)

        # Pooling, flattening, dropout, and final classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, in_channels, red1x1, out1x1, red3x3, out3x3, pool_proj):
        super(InceptionModule, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, red1x1, kernel_size=1),
            nn.BatchNorm2d(red1x1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red1x1, out1x1, kernel_size=1),
            nn.BatchNorm2d(out1x1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red3x3, kernel_size=1),
            nn.BatchNorm2d(red3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(red3x3, out3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out3x3),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        outputs = [b1, b2, b3]
        return torch.cat(outputs, 1)