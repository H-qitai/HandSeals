import torch
import torch.nn as nn
import torch.nn.functional as F

# Definition of a BasicBlock used in ResNet, handling the residual connections
class BasicBlock(nn.Module):
    expansion = 1  # This factor scales the number of output features of the block

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # Batch normalization after the first convolution
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # Batch normalization after the second convolution
        self.downsample = downsample  # Downsample function to match dimensions if needed

    def forward(self, x):
        identity = x  # Save input for residual connection

        # First convolutional operation followed by ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second convolutional operation
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsampling if defined
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the residual
        out += identity
        out = F.relu(out)  # Final ReLU after adding the residual
        return out

# ResNet model definition
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # Initial convolutional layer with larger receptive field
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stacked layers using the block definitions
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive average pooling to prepare for the fully connected layer
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # Final fully connected layer

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # Determine if downsampling is needed to adjust dimensions
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # Construct the layer by stacking blocks
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolutional and pooling operations
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Process through each residual block layer
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pool, flatten, and pass through the fully connected layer to get class scores
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Example instantiation of ResNet-34
model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=1000)
