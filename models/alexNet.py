import torch
import torch.nn as nn
import torch.nn.functional as F

class   AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolutional layer with 32 output channels, kernel size 3, stride 1, and padding 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # First pooling layer
            
            # Second convolutional layer with 64 output channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Second pooling layer
        )
        
        # Adaptive average pooling layer to reduce each feature map to 6x6
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 6 * 6, 1024),  # Fully connected layer with 1024 units
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),  # Final fully connected layer that outputs class probabilities
        )

    def forward(self, x):
        x = self.features(x)  # Apply the feature extraction layers
        x = self.avgpool(x)  # Apply adaptive average pooling
        x = torch.flatten(x, 1)  # Flatten the output for the fully connected layers
        x = self.classifier(x)  # Apply the classifier
        return x