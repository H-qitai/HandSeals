import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        # Feature extractor part of AlexNet
        self.features = nn.Sequential(
            # First convolutional layer with 96 output channels, 11x11 kernel size, stride of 4, and padding of 2
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Second convolutional layer with 256 output channels, 5x5 kernel size, and padding of 2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Third convolutional layer with 384 output channels, 3x3 kernel size, and padding of 1
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Fourth convolutional layer, same as third
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Fifth convolutional layer with 256 output channels, 3x3 kernel size, and padding of 1
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Final max pooling layer with a 3x3 kernel size and a stride of 2
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Dropout layer to prevent overfitting by randomly setting a fraction of input units to 0 during training
        self.dropout = nn.Dropout()
        
        # Classifier part of AlexNet
        self.classifier = nn.Sequential(
            # First fully connected layer with 4096 output units, taking input from feature layer
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # Dropout layer again to prevent overfitting
            nn.Dropout(),
            # Second fully connected layer with 4096 output units
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # Final fully connected layer that outputs to the number of classes
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        # Pass input through the feature extractor part
        x = self.features(x)
        # Flatten the output from feature extractor to fit the fully connected layer
        x = torch.flatten(x, 1)
        # Apply dropout
        x = self.dropout(x)
        # Pass through the classifier part
        x = self.classifier(x)
        return x
