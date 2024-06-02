import torch, cv2
from models.alexNet import AlexNet
from models.resNet50 import ResNet, BasicBlock
from models.inceptionV1 import InceptionV1
from models.DenseNet import DenseNet
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt

def load_model(filename):
    # Use torch.device to load the model onto GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model with map_location parameter set to device
    state = torch.load(filename, map_location=device)
    
    config = state['config']
    
    # Initialize the model based on the saved configuration
    if config['model_name'] == 'AlexNet':
        model = AlexNet(num_classes=36)
    elif config['model_name'] == 'ResNet':
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=36)
    elif config['model_name'] == 'InceptionV1':
        model = InceptionV1(num_classes=36)
    elif config['model_name'] == 'DenseNet':
        model = DenseNet(num_classes=36)
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")

    model.load_state_dict(state['model_state_dict'])
    model.to(device)  # Ensure model is on the correct device
    model.eval()
    
    return model, config


class DetailedViewWindow(QDialog):
    def __init__(self, image, probabilities, predicted_label, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Image Prediction')
        self.layout = QVBoxLayout(self)

        # Convert grayscale image to RGB if necessary
        if len(image.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Display the image
        height, width, channels = image.shape
        bytes_per_line = channels * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        image_label = QLabel()
        image_label.setPixmap(pixmap.scaled(420, 420, Qt.KeepAspectRatio))

        # Center the image using QHBoxLayout
        image_layout = QHBoxLayout()
        image_layout.addStretch(1)
        image_layout.addWidget(image_label)
        image_layout.addStretch(1)

        self.layout.addLayout(image_layout)

        # Display the predicted result
        predicted_label_widget = QLabel(f'Predicted Result: {predicted_label}')
        predicted_label_widget.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(predicted_label_widget)

        # Display the output probabilities
        self.figure = plt.figure(figsize=(10, 3))  # Set the size of the figure
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.plot_probabilities(probabilities)

    def plot_probabilities(self, probabilities):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.bar(range(len(probabilities)), probabilities)
        ax.set_xlabel('Class')
        ax.set_ylabel('Probability')
        ax.set_title('Output Probabilities')
        ax.set_xticks(range(36))  # Ensure all classes from 0 to 35 are shown
        self.figure.tight_layout(pad=2.0)  # Add padding to ensure labels are not cut off
        self.canvas.draw()

