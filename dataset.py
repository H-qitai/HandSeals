from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class HandSealDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale if needed
            transforms.Resize((96, 96)),  # Resize images to a larger size
            transforms.ToTensor(),  # Converts to FloatTensor and scales the values to [0, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label, image = self.images[idx]
        if isinstance(image, np.ndarray):
            if image.shape[0] == 1:  # Check if the first dimension is 1
                image = image.squeeze(0)  # Remove the first dimension
            image = Image.fromarray(image)
        image = self.transform(image)  # Apply the transformation
        return image, label