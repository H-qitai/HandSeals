import sys
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar, QFileDialog, QScrollArea, QLabel, QGridLayout, QLineEdit, QComboBox, QSlider, QStackedWidget
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
# from resNet50 import ResNet, BasicBlock
from dataloader import CSVDataLoader
from training import TrainingThreadResnet, TrainingThreadInceptionV1, TrainingThreadAlexNet
from dataset import HandSealDataset
# from inceptionV1 import InceptionV1
from torchvision import transforms

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(fig)

class HandSeals(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Hand Seals')
        self.resize(1000, 800)
        self.images = []
        self.filtered_images = []
        self.current_image_index = 0
        self.data_loader = None
        self.search_bar_added = False
        self.training_thread = None
        self.initUI()

    def initUI(self):
        self.vertical_layout = QVBoxLayout(self)
        self.horizontal_layout = QHBoxLayout()

        # Buttons
        self.init_buttons()

        # Progress Bar and Time Left
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.horizontal_layout.addWidget(self.progress_bar)
        self.time_left_label = QLabel("Time left: 00:00")
        self.horizontal_layout.addWidget(self.time_left_label)

        # Stacked widget to switch between different views
        self.stacked_widget = QStackedWidget()
        self.vertical_layout.addLayout(self.horizontal_layout)
        self.vertical_layout.addWidget(self.stacked_widget)

        # Create view page
        self.view_page = QWidget()
        self.view_layout = QVBoxLayout(self.view_page)

        # Search bar
        self.search_bar = QLineEdit()  # Initialize the search bar here
        self.search_bar.setPlaceholderText("Search by label...")
        self.search_bar.textChanged.connect(self.filter_images)

        # Image Scroll Area
        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidgetResizable(True)
        self.image_container = QWidget()
        self.image_layout = QGridLayout(self.image_container)
        self.image_scroll_area.setWidget(self.image_container)

        self.view_layout.addWidget(self.search_bar)  # Now this line will work
        self.view_layout.addWidget(self.image_scroll_area)

        self.stacked_widget.addWidget(self.view_page)

        # Create train page
        self.train_page = QWidget()
        self.train_layout = QVBoxLayout(self.train_page)
        self.train_settings()

        self.stacked_widget.addWidget(self.train_page)

        # Training progress page
        self.train_progress_page = QWidget()
        self.train_progress_layout = QVBoxLayout(self.train_progress_page)
        self.train_progress_bar = QProgressBar()
        self.train_progress_layout.addWidget(self.train_progress_bar)

        # Labels for epoch, batch, and accuracy
        self.epoch_label = QLabel("Epoch: 0/0")
        self.train_progress_layout.addWidget(self.epoch_label)

        self.batch_label = QLabel("Batch: 0/0")
        self.train_progress_layout.addWidget(self.batch_label)

        self.accuracy_label = QLabel("Accuracy: 0%")
        self.train_progress_layout.addWidget(self.accuracy_label)

        # Create a horizontal layout for the graphs
        self.graph_layout = QHBoxLayout()

        # Add Matplotlib plots
        self.val_accuracy_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.graph_layout.addWidget(self.val_accuracy_canvas)

        self.train_loss_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.graph_layout.addWidget(self.train_loss_canvas)

        # Add the horizontal graph layout to the train progress layout
        self.train_progress_layout.addLayout(self.graph_layout)

        self.stop_training_button = QPushButton('Stop Training')
        self.stop_training_button.clicked.connect(self.stop_training)
        self.train_progress_layout.addWidget(self.stop_training_button)

        self.stacked_widget.addWidget(self.train_progress_page)

        # Show the initial view page
        self.stacked_widget.setCurrentWidget(self.view_page)

        # Connect the scroll event
        self.image_scroll_area.verticalScrollBar().valueChanged.connect(self.check_scroll)


    def init_buttons(self):
        self.load_data_button = QPushButton('Load Data')
        self.load_data_button.clicked.connect(self.button1_clicked)
        self.horizontal_layout.addWidget(self.load_data_button)

        self.view_button = QPushButton('View')
        self.view_button.clicked.connect(self.show_view_page)
        self.horizontal_layout.addWidget(self.view_button)

        self.train_button = QPushButton('Train')
        self.train_button.clicked.connect(self.show_train_page)
        self.horizontal_layout.addWidget(self.train_button)

        self.test_button = QPushButton('Test')
        self.test_button.clicked.connect(self.button4_clicked)
        self.horizontal_layout.addWidget(self.test_button)

        self.stop_button = QPushButton('Stop Loading')
        self.stop_button.clicked.connect(self.stop_loading)
        self.stop_button.setEnabled(False)  # Initially disabled
        self.horizontal_layout.addWidget(self.stop_button)

    def button1_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.load_data(file_path)

    def load_data(self, file_path):
        self.data_loader = CSVDataLoader(file_path)
        self.data_loader.dataLoaded.connect(self.on_data_loaded)
        self.data_loader.loadError.connect(self.on_data_load_error)
        self.data_loader.progressUpdated.connect(self.progress_bar.setValue)
        self.data_loader.timeLeftUpdated.connect(self.update_time_left)
        self.stop_button.setEnabled(True)  # Enable stop button when loading starts
        self.data_loader.start()

    def stop_loading(self):
        self.data_loader.stop()
        self.progress_bar.setValue(0)
        self.time_left_label.setText("Time left: 00:00")
        self.stop_button.setEnabled(False)  # Disable stop button when loading stops
        if not self.images:  # Disable start training button if no images are loaded
            self.start_training_button.setEnabled(False)

    def on_data_loaded(self, images):
        self.images = images
        self.filtered_images = images
        self.progress_bar.setValue(100)
        self.stop_button.setEnabled(False)  # Disable stop button when loading completes
        if self.images:  # Enable start training button if images are loaded
            self.start_training_button.setEnabled(True)

    def on_data_load_error(self, error):
        print(f"Error loading data: {error}")
        self.stop_button.setEnabled(False)  # Disable stop button if loading encounters an error
        self.start_training_button.setEnabled(False) 

    def update_time_left(self, minutes, seconds):
        self.time_left_label.setText(f"Time left: {minutes:02}:{seconds:02}")

    def show_view_page(self):
        self.stacked_widget.setCurrentWidget(self.view_page)
        if self.images:
            self.current_image_index = 0
            self.clear_layout(self.image_layout)
            self.add_images_incrementally()

    def add_images_incrementally(self):
        batch_size = 100
        start_index = self.current_image_index
        end_index = min(start_index + batch_size, len(self.filtered_images))
        for i in range(start_index, end_index):
            label, pixels = self.filtered_images[i]
            pixmap = self.array_to_pixmap(pixels)
            label_widget = QLabel()
            label_widget.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio))
            self.image_layout.addWidget(label_widget, i // 10, i % 10)
        self.current_image_index = end_index

    def array_to_pixmap(self, array):
        if len(array.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        elif len(array.shape) == 3 and array.shape[0] == 1:  # Grayscale image with channel dimension
            image = cv2.cvtColor(array[0], cv2.COLOR_GRAY2RGB)
        elif len(array.shape) == 3 and array.shape[0] == 3:  # RGB image
            image = array.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        else:
            raise ValueError("Invalid image shape")

        bytes_per_line = 3 * image.shape[1]
        q_image = QImage(image.data, image.shape[1], image.shape[0], bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)

    def check_scroll(self, value):
        if value == self.image_scroll_area.verticalScrollBar().maximum():
            if self.current_image_index < len(self.filtered_images):
                self.add_images_incrementally()

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def filter_images(self):
        search_text = self.search_bar.text().strip().lower()
        if search_text:
            self.filtered_images = [img for img in self.images if search_text in str(img[0]).lower()]
        else:
            self.filtered_images = self.images
        self.current_image_index = 0
        self.clear_layout(self.image_layout)
        self.add_images_incrementally()

    def show_train_page(self):
        self.stacked_widget.setCurrentWidget(self.train_page)

    def update_train_test_ratio_label(self, value):
        self.train_test_ratio_value_label.setText(str(value))

    def update_batch_size_label(self, value):
        self.batch_size_value_label.setText(str(value))

    def update_epochs_label(self, value):
        self.epochs_value_label.setText(str(value))

    def train_settings(self):
        font = QFont("Arial", 10, QFont.Bold)

        self.start_training_button = QPushButton("Start Training")
        self.start_training_button.clicked.connect(self.start_training)
        self.start_training_button.setEnabled(False)
        self.train_layout.addWidget(self.start_training_button)

        self.train_layout.addSpacing(30)

        self.model_label = QLabel("Select Model")
        self.model_label.setAlignment(Qt.AlignCenter)
        self.model_label.setFont(font)
        self.train_layout.addWidget(self.model_label)

        self.train_layout.addSpacing(20)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["AlexNet", "ResNet", "Inception V1"])
        self.train_layout.addWidget(self.model_combo)

        self.train_layout.addSpacing(60)

        self.train_test_ratio_label = QLabel("Train/Test Ratio")
        self.train_test_ratio_label.setAlignment(Qt.AlignCenter)
        self.train_test_ratio_label.setFont(font)
        self.train_layout.addWidget(self.train_test_ratio_label)

        self.train_layout.addSpacing(20)

        self.train_test_ratio_slider = QSlider(Qt.Horizontal)
        self.train_test_ratio_slider.setMinimum(1)
        self.train_test_ratio_slider.setMaximum(99)
        self.train_test_ratio_slider.setValue(80)
        self.train_test_ratio_slider.setMinimumHeight(30)
        self.train_test_ratio_slider.setMinimumWidth(200)
        self.train_test_ratio_slider.valueChanged.connect(self.update_train_test_ratio_label)
        self.train_layout.addWidget(self.train_test_ratio_slider)

        self.train_test_ratio_value_label = QLabel("80")
        self.train_test_ratio_value_label.setAlignment(Qt.AlignCenter)
        self.train_layout.addWidget(self.train_test_ratio_value_label)

        self.train_layout.addSpacing(60)

        self.batch_size_label = QLabel("Batch Size")
        self.batch_size_label.setAlignment(Qt.AlignCenter)
        self.batch_size_label.setFont(font)
        self.train_layout.addWidget(self.batch_size_label)

        self.train_layout.addSpacing(20)

        self.batch_size_slider = QSlider(Qt.Horizontal)
        self.batch_size_slider.setMinimum(25)
        self.batch_size_slider.setMaximum(200)
        self.batch_size_slider.setValue(100)
        self.batch_size_slider.setMinimumHeight(30)
        self.batch_size_slider.setMinimumWidth(200)
        self.batch_size_slider.valueChanged.connect(self.update_batch_size_label)
        self.train_layout.addWidget(self.batch_size_slider)

        self.batch_size_value_label = QLabel("100")
        self.batch_size_value_label.setAlignment(Qt.AlignCenter)
        self.train_layout.addWidget(self.batch_size_value_label)

        self.train_layout.addSpacing(60)

        self.epochs_label = QLabel("Epochs")
        self.epochs_label.setAlignment(Qt.AlignCenter)
        self.epochs_label.setFont(font)
        self.train_layout.addWidget(self.epochs_label)

        self.train_layout.addSpacing(20)

        self.epochs_slider = QSlider(Qt.Horizontal)
        self.epochs_slider.setMinimum(2)
        self.epochs_slider.setMaximum(100)
        self.epochs_slider.setValue(30)
        self.epochs_slider.setMinimumHeight(30)
        self.epochs_slider.setMinimumWidth(200)
        self.epochs_slider.valueChanged.connect(self.update_epochs_label)
        self.train_layout.addWidget(self.epochs_slider)

        self.epochs_value_label = QLabel("30")
        self.epochs_value_label.setAlignment(Qt.AlignCenter)
        self.train_layout.addWidget(self.epochs_value_label)

        self.train_layout.addSpacing(60)

    def start_training(self):

        selected_model = self.model_combo.currentText()
        batch_size = self.batch_size_slider.value()
        epochs = self.epochs_slider.value()
        train_test_ratio = self.train_test_ratio_slider.value() / 100.0

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale if needed
            transforms.Resize((28, 28)),
            transforms.ToTensor(),  # Converts to FloatTensor and scales the values to [0, 1]
        ])

        dataset = HandSealDataset(self.images, transform=transform)  # Use HandSealDataset for images

        # Ensure there are enough samples for training and testing
        if len(dataset) == 0:
            print("Dataset is empty. Please load valid data.")
            return

        train_size = int(len(dataset) * train_test_ratio)
        test_size = len(dataset) - train_size
        if train_size == 0 or test_size == 0:
            print("Insufficient data for training and testing. Adjust the train/test ratio or load more data.")
            return

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if selected_model == 'AlexNet':
            self.train_alexnet(train_loader, test_loader, epochs)
        elif selected_model == 'ResNet':
            self.train_resnet(train_loader, test_loader, epochs)
        elif selected_model == 'Inception V1':
            self.train_inceptionV1(train_loader, test_loader, epochs)

    def update_train_loss_plot(self, train_losses):
        self.train_loss_canvas.ax.clear()
        self.train_loss_canvas.ax.plot(range(len(train_losses)), train_losses, 'r-')
        self.train_loss_canvas.ax.set_title('Training Loss', fontsize=14)
        self.train_loss_canvas.ax.set_xlabel('Epoch', fontsize=10)
        self.train_loss_canvas.ax.set_ylabel('Loss', fontsize=10)
        self.val_accuracy_canvas.ax.grid(True)  # Add grid
        self.train_loss_canvas.draw()

    def update_val_accuracy_plot(self, val_accuracies):
        self.val_accuracy_canvas.ax.clear()
        self.val_accuracy_canvas.ax.plot(range(len(val_accuracies)), val_accuracies, 'b-')
        self.val_accuracy_canvas.ax.set_title('Validation Accuracy Over Epochs', fontsize=14)
        self.val_accuracy_canvas.ax.set_xlabel('Epoch', fontsize=10)
        self.val_accuracy_canvas.ax.set_ylabel('Accuracy (%)', fontsize=10)
        self.val_accuracy_canvas.ax.grid(True)  # Add grid
        self.val_accuracy_canvas.draw()

        # Update the accuracy label with the latest accuracy
        if val_accuracies:
            self.accuracy_label.setText(f"Accuracy: {val_accuracies[-1]:.2f}%")

    def update_epoch_label(self, epoch, loss):
        self.epoch_label.setText(f"Epoch: {epoch}/{self.epochs_slider.value()} - Loss: {loss:.4f}")

    def update_batch_label(self, epoch, batch, loss):
        self.batch_label.setText(f"Epoch: {epoch} - Batch: {batch}/{len(self.train_loader)} - Loss: {loss:.4f}")

    def stop_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.train_progress_bar.setValue(0)
            self.epoch_label.setText("Epoch: 0/0")
            self.batch_label.setText("Batch: 0/0")
        else:
            print("No training in progress")

    def on_training_stopped(self):
        print("Training Stopped")
        # Ensure that the train progress page is still visible
        self.stacked_widget.setCurrentWidget(self.train_progress_page)

        # Disable the stop training button since training has stopped
        self.stop_training_button.setDisabled(True)

        # Re-enable the start training button for possible new training sessions
        self.start_training_button.setEnabled(True)

        # Ensure the progress bar reflects the complete state if training finished normally
        self.train_progress_bar.setValue(100)

        # Final updates to the plots to ensure they show the last state
        self.update_train_loss_plot(self.training_thread.final_train_losses)
        self.update_val_accuracy_plot(self.training_thread.final_val_accuracies)

    def train_alexnet(self, train_loader, test_loader, epochs):
        self.stacked_widget.setCurrentWidget(self.train_progress_page)
        self.training_thread = TrainingThreadAlexNet(train_loader, test_loader, epochs)
        self.train_loader = train_loader
        self.training_thread.progress_updated.connect(self.train_progress_bar.setValue)
        self.training_thread.training_stopped.connect(self.on_training_stopped)
        self.training_thread.epoch_updated.connect(self.update_epoch_label)
        self.training_thread.batch_updated.connect(self.update_batch_label)
        self.training_thread.train_loss_updated.connect(self.update_train_loss_plot)
        self.training_thread.val_accuracy_updated.connect(self.update_val_accuracy_plot)
        self.training_thread.start()

    def train_inceptionV1(self, train_loader, test_loader, epochs):
        self.stacked_widget.setCurrentWidget(self.train_progress_page)
        self.training_thread = TrainingThreadInceptionV1(train_loader, test_loader, epochs)
        self.train_loader = train_loader
        self.training_thread.progress_updated.connect(self.train_progress_bar.setValue)
        self.training_thread.training_stopped.connect(self.on_training_stopped)
        self.training_thread.epoch_updated.connect(self.update_epoch_label)
        self.training_thread.batch_updated.connect(self.update_batch_label)
        self.training_thread.train_loss_updated.connect(self.update_train_loss_plot)
        self.training_thread.val_accuracy_updated.connect(self.update_val_accuracy_plot)
        self.training_thread.start()

    def train_resnet(self, train_loader, test_loader, epochs):
        self.stacked_widget.setCurrentWidget(self.train_progress_page)
        self.training_thread = TrainingThreadResnet(train_loader, test_loader, epochs)
        self.train_loader = train_loader
        self.training_thread.progress_updated.connect(self.train_progress_bar.setValue)
        self.training_thread.training_stopped.connect(self.on_training_stopped)
        self.training_thread.epoch_updated.connect(self.update_epoch_label)
        self.training_thread.batch_updated.connect(self.update_batch_label)
        self.training_thread.train_loss_updated.connect(self.update_train_loss_plot)
        self.training_thread.val_accuracy_updated.connect(self.update_val_accuracy_plot)
        self.training_thread.start()

    def button4_clicked(self):
        print("Testing...")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HandSeals()
    window.show()
    sys.exit(app.exec_())
