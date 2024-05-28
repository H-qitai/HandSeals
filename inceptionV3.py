import torch
import torch.nn as nn
import torch.nn.functional as F

# Definition of the Inception V3 model
class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionV3, self).__init__()
        # Initial convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)  # First conv layer with stride 2 for downsampling
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1)  # Second conv layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Third conv layer with padding to maintain size
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # Max pooling for spatial reduction

        # Inception modules with reduced parameters for simplicity
        self.inception1 = InceptionModule(64, 64, 96, 64, 96, 32)  # First inception module
        self.inception2 = InceptionModule(128, 128, 192, 128, 192, 64)  # Second inception module

        # Final classifier components
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive average pooling to output a fixed size
        self.dropout = nn.Dropout(0.4)  # Dropout for regularization
        self.fc = nn.Linear(448, num_classes)  # Fully connected layer for classification

    def forward(self, x):
        # Apply initial convolutional layers with ReLU activations
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.pool1(x)  # Apply max pooling

        # Apply inception modules
        x = self.inception1(x)
        x = self.inception2(x)

        # Apply average pooling, flatten, dropout, and fully connected layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Definition of an individual Inception module
class InceptionModule(nn.Module):
    def __init__(self, in_channels, red1x1, out1x1, red3x3, out3x3, pool_proj):
        super(InceptionModule, self).__init__()
        # Branch 1: 1x1 convolution reducing dimensions, then outputting more channels
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, red1x1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red1x1, out1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # Branch 2: Reduce dimensions, then expand with a 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red3x3, out3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Branch 3: Pooling followed by a projection with 1x1 convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Apply each branch in parallel
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        b3 = self.branch3(b3)
        # Concatenate all branch outputs along the channel dimension
        outputs = [b1, b2, b3]
        return torch.cat(outputs, 1)

