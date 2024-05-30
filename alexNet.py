import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):  # Initialize the AlexNet model with a configurable number of output classes
        super(AlexNet, self).__init__()  # Call the initializer of the parent nn.Module class
        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolutional layer with 64 output channels, kernel size 3, stride 1, and padding 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  # Activation function to introduce non-linearity (in-place to save memory)
            nn.MaxPool2d(kernel_size=2, stride=2),  # First pooling layer to reduce spatial dimensions
            
            # Second convolutional layer with 192 output channels, using same padding to maintain feature map size
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Second pooling layer

            # Third convolutional layer increasing to 384 channels, no pooling here to preserve feature map size
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # Activation function

            # Fourth convolutional layer with 256 channels
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # Activation function

            # Fifth convolutional layer also with 256 channels
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2),  # Third and final pooling layer
        )
        
        # Adaptive average pooling layer to reduce each feature map to 6x6
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(),  # Dropout layer to prevent overfitting by randomly zeroing some of the elements
            nn.Linear(256 * 6 * 6, 4096),  # Fully connected layer with 4096 units
            nn.ReLU(inplace=True),  # Activation function
            nn.Dropout(),  # Another dropout layer
            nn.Linear(4096, 4096),  # Another fully connected layer with 4096 units
            nn.ReLU(inplace=True),  # Activation function
            nn.Linear(4096, num_classes),  # Final fully connected layer that outputs class probabilities
        )

    def forward(self, x):
        # Forward pass definition
        x = self.features(x)  # Apply the feature extraction layers
        x = self.avgpool(x)  # Apply adaptive average pooling
        x = torch.flatten(x, 1)  # Flatten the output for the fully connected layers
        x = self.classifier(x)  # Apply the classifier
        return x  # Return the output

### Initialize the AlexNet model for the number of classes you have (e.g., 10 for MNIST) ###
### model = AlexNet(num_classes=10) ###