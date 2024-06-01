import sys, time, cv2, random, torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QDialog, QWidget, QVBoxLayout, QPushButton, QProgressBar, QFileDialog, QScrollArea, QLabel, QGridLayout, QLineEdit, QComboBox, QSlider, QStackedWidget, QMessageBox, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from dataloader import CSVDataLoader
from training import TrainingThreadResnet, TrainingThreadInceptionV1, TrainingThreadAlexNet, TrainingThreadDenseNet
from dataset import HandSealDataset
from utils.model_utils import load_model, DetailedViewWindow

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(fig)

class HandSeals(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Hand Seals')
        self.resize(1000, 800)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define a dictionary to remap the labels
        self.label_remap = {
            26: 35,
            27: 26,
            28: 33,
            29: 32,
            30: 27,
            31: 34,
            32: 30,
            33: 29,
            34: 28,
            35: 31
        }

        self.images = []
        self.filtered_images = []
        self.current_image_index = 0
        self.data_loader = None
        self.search_bar_added = False
        self.training_thread = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.start_time = 0
        
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

        # Add a page for displaying test results
        self.test_results_page = QWidget()
        self.test_results_layout = QVBoxLayout(self.test_results_page)

        # Search bar for test results
        self.test_search_bar = QLineEdit()
        self.test_search_bar.setPlaceholderText("Search test results by label...")
        self.test_search_bar.textChanged.connect(self.filter_test_images)
        self.test_results_layout.addWidget(self.test_search_bar)

        self.test_image_scroll_area = QScrollArea()
        self.test_image_scroll_area.setWidgetResizable(True)
        self.test_image_container = QWidget()
        self.test_image_layout = QGridLayout(self.test_image_container)
        self.test_image_scroll_area.setWidget(self.test_image_container)
        self.test_results_layout.addWidget(self.test_image_scroll_area)

        self.stacked_widget.addWidget(self.test_results_page)

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

        self.timer_label = QLabel("Time Elapsed: 00:00:00", self)
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.train_progress_layout.addWidget(self.timer_label)

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
        self.test_image_scroll_area.verticalScrollBar().valueChanged.connect(self.check_test_scroll)

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
        self.test_button.clicked.connect(self.load_and_test_model)
        self.horizontal_layout.addWidget(self.test_button)

        self.stop_button = QPushButton('Stop Loading')
        self.stop_button.clicked.connect(self.stop_loading)
        self.stop_button.setEnabled(False)  # Initially disabled
        self.horizontal_layout.addWidget(self.stop_button)

    def load_and_test_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "Model Files (*.pth);;All Files (*)", options=options)
        if not file_path:
            return

        try:
            # Load the model and configuration
            model, config = load_model(file_path)
            model.to(self.device)  # Send the model to the appropriate device (CPU/GPU)

            # Create the test DataLoader based on the configuration
            dataset = HandSealDataset(self.images)  # Use HandSealDataset for images

            train_size = int(len(dataset) * config['train_test_ratio'])
            test_size = len(dataset) - train_size

            if test_size == 0:
                QMessageBox.critical(self, "Error", "Test dataset is empty. Cannot proceed with testing.")
                return

            _, test_dataset = random_split(dataset, [train_size, test_size])
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

            # Ensure test_loader is not empty
            if len(test_loader) == 0:
                QMessageBox.critical(self, "Error", "Test DataLoader is empty. Cannot proceed with testing.")
                return

            # Evaluate the model on the test dataset
            model.eval()
            correct = 0
            total = 0
            self.test_images = []
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    probabilities = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    # Convert tensor to numpy array and denormalize if necessary
                    for img, label, pred, prob in zip(images.cpu().numpy(), labels.cpu().numpy(), predicted.cpu().numpy(), probabilities.cpu().numpy()):
                        img = (img * 255).astype(np.uint8)  # Convert from [0, 1] range to [0, 255]
                        if img.shape[0] == 1:  # If image is grayscale with channel dimension
                            img = img[0]
                        img = img.transpose(1, 2, 0) if img.shape[0] == 3 else img  # Convert from (C, H, W) to (H, W, C) if needed
                        self.test_images.append((label, pred, img, prob))

            # Prevent division by zero
            if total == 0:
                accuracy = 0.0
            else:
                accuracy = 100 * correct / total

            self.show_test_results_page()
            QMessageBox.information(self, "Model Accuracy", f"Accuracy of the model on the test dataset: {accuracy:.2f}%")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load and test the model:\n{str(e)}")



    # file select
    def button1_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.load_data(file_path)

    # Load data from a CSV file, and update the progress bar
    def load_data(self, file_path):
        self.data_loader = CSVDataLoader(file_path)
        self.data_loader.dataLoaded.connect(self.on_data_loaded)
        self.data_loader.loadError.connect(self.on_data_load_error)
        self.data_loader.progressUpdated.connect(self.progress_bar.setValue)
        self.data_loader.timeLeftUpdated.connect(self.update_time_left)
        self.stop_button.setEnabled(True)  # Enable stop button when loading starts
        self.data_loader.start()

    # Stop loading the data
    def stop_loading(self):
        self.data_loader.stop()
        self.progress_bar.setValue(0)
        self.time_left_label.setText("Time left: 00:00")
        self.stop_button.setEnabled(False)  # Disable stop button when loading stops
        if not self.images:  # Disable start training button if no images are loaded
            self.start_training_button.setEnabled(False)

    # Load the data into the application, and remap the labels
    def on_data_loaded(self, images):
        # Remap the labels
        remapped_images = []
        for label, pixels in images:
            if label in self.label_remap:
                remapped_label = self.label_remap[label]
            else:
                remapped_label = label
            remapped_images.append((remapped_label, pixels))

        self.images = remapped_images
        self.filtered_images = self.images
        self.progress_bar.setValue(100)
        self.stop_button.setEnabled(False)
        if self.images:
            self.start_training_button.setEnabled(True)

    # Error handling
    def on_data_load_error(self, error):
        print(f"Error loading data: {error}")
        self.stop_button.setEnabled(False)  # Disable stop button if loading encounters an error
        self.start_training_button.setEnabled(False) 

    # Update the time left
    def update_time_left(self, minutes, seconds):
        self.time_left_label.setText(f"Time left: {minutes:02}:{seconds:02}")

    # Show the view page
    def show_view_page(self):
        self.stacked_widget.setCurrentWidget(self.view_page)
        if self.images:
            self.current_image_index = 0
            self.clear_layout(self.image_layout)
            self.add_images_incrementally()

    def show_test_results_page(self):
        self.stacked_widget.setCurrentWidget(self.test_results_page)
        self.current_image_index = 0
        self.clear_layout(self.test_image_layout)
        self.filtered_test_images = self.test_images 
        self.add_filtered_test_images_incrementally()

    # Add images to the layout incrementally to prevent lag
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

    # Add filtered test images incrementally
    def add_filtered_test_images_incrementally(self):
        batch_size = 100
        start_index = self.current_image_index
        end_index = min(start_index + batch_size, len(self.filtered_test_images))
        for i in range(start_index, end_index):
            label, predicted, img_data, probabilities = self.filtered_test_images[i]
            pixmap = self.array_to_pixmap(img_data)
            image_button = QPushButton()
            image_button.setIcon(QIcon(pixmap))
            image_button.setIconSize(QSize(100, 100))
            image_button.clicked.connect(lambda _, img=img_data, probs=probabilities, pred=predicted: self.show_detailed_view(img, probs, pred))
            self.test_image_layout.addWidget(image_button, i // 10, i % 10)
        self.current_image_index = end_index


    def show_detailed_view(self, image, probabilities, predicted_label):
        self.detailed_view = DetailedViewWindow(image, probabilities, predicted_label)
        self.detailed_view.exec_()

    # Convert an image array to a QPixmap
    def array_to_pixmap(self, array):
        if len(array.shape) == 2:  # Grayscale image
            image = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        elif len(array.shape) == 3 and array.shape[0] == 1:  # Grayscale image with channel dimension
            image = cv2.cvtColor(array[0], cv2.COLOR_GRAY2RGB)
        elif len(array.shape) == 3 and array.shape[0] == 3:  # RGB image (C, H, W)
            image = array.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        elif len(array.shape) == 3 and array.shape[2] == 3:  # RGB image (H, W, C)
            image = array
        else:
            raise ValueError("Invalid image shape")

        height, width, _ = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)


    # Filter images based on the search bar
    def check_scroll(self, value):
        if value == self.image_scroll_area.verticalScrollBar().maximum():
            if self.current_image_index < len(self.filtered_images):
                self.add_images_incrementally()

    def check_test_scroll(self, value):
        if value == self.test_image_scroll_area.verticalScrollBar().maximum():
            if self.current_image_index < len(self.filtered_test_images):
                self.add_filtered_test_images_incrementally()

    # Clear the layout
    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    # Ensure your filter_images function filters based on mapped labels
    def filter_images(self):
        search_text = self.search_bar.text().strip()
        if search_text.isdigit():
            label = int(search_text)
            self.filtered_images = [img for img in self.images if img[0] == label]
        else:
            self.filtered_images = self.images

        self.current_image_index = 0
        self.clear_layout(self.image_layout)
        self.add_images_incrementally()

    # Filter test images based on the search bar
    def filter_test_images(self):
        search_text = self.test_search_bar.text().strip()
        if search_text.isdigit():
            label = int(search_text)
            self.filtered_test_images = [img for img in self.test_images if img[0] == label]
        else:
            self.filtered_test_images = self.test_images

        self.current_image_index = 0
        self.clear_layout(self.test_image_layout)
        self.add_filtered_test_images_incrementally()

    def show_train_page(self):
        self.stacked_widget.setCurrentWidget(self.train_page)

    def update_train_test_ratio_label(self, value):
        self.train_test_ratio_value_label.setText(str(value))

    def update_batch_size_label(self, value):
        self.batch_size_value_label.setText(str(value))

    def update_epochs_label(self, value):
        self.epochs_value_label.setText(str(value))

    # train page
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
        self.model_combo.addItems(["AlexNet", "ResNet", "Inception V1", "DenseNet"])
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

    # Training, configure the hyper parameters, and select which model and start training.
    def start_training(self):

        self.reset_timer()
        self.start_time = time.time()
        self.timer.start(1000)  # Update every second

        selected_model = self.model_combo.currentText()
        batch_size = self.batch_size_slider.value()
        epochs = self.epochs_slider.value()
        train_test_ratio = self.train_test_ratio_slider.value() / 100.0

        dataset = HandSealDataset(self.images)  # Use HandSealDataset for images

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
        elif selected_model == 'DenseNet':
            self.train_densenet(train_loader, test_loader, epochs)

        # Re-enable the Stop Training button when starting a new training session
        self.stop_training_button.setEnabled(True)

    # Update the graph when training
    def update_train_loss_plot(self, train_losses):
        self.train_loss_canvas.ax.clear()
        self.train_loss_canvas.ax.plot(range(len(train_losses)), train_losses, 'r-')
        self.train_loss_canvas.ax.set_title('Training Loss', fontsize=14)
        self.train_loss_canvas.ax.set_xlabel('Epoch', fontsize=10)
        self.train_loss_canvas.ax.set_ylabel('Loss', fontsize=10)
        self.val_accuracy_canvas.ax.grid(True)  # Add grid
        self.train_loss_canvas.draw()

    # Update the graph when training
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

    # Text information for the training progress
    def update_epoch_label(self, epoch, loss):
        self.epoch_label.setText(f"Epoch: {epoch}/{self.epochs_slider.value()} - Loss: {loss:.4f}")

    def update_batch_label(self, epoch, batch, loss):
        self.batch_label.setText(f"Epoch: {epoch} - Batch: {batch}/{len(self.train_loader)} - Loss: {loss:.4f}")

    # Timer for training
    def update_timer(self):
        elapsed_time = int(time.time() - self.start_time)
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.timer_label.setText(f"Time Elapsed: {hours:02}:{minutes:02}:{seconds:02}")

    def reset_timer(self):
        self.timer_label.setText("Time Elapsed: 00:00:00")
        self.start_time = 0

    def stop_training(self):
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.stop()
            self.training_thread.wait()
            self.stop_training_button.setEnabled(False)
            self.timer.stop()
            print("Training stopped.")

    def on_training_stopped(self):
        print("Training Stopped")
        self.stop_training_button.setEnabled(False)
        self.timer.stop()
        # Ensure that the train progress page is still visible
        self.stacked_widget.setCurrentWidget(self.train_progress_page)

        # Ensure the progress bar reflects the complete state if training finished normally
        self.train_progress_bar.setValue(100)

        # Final updates to the plots to ensure they show the last state
        self.update_train_loss_plot(self.training_thread.final_train_losses)
        self.update_val_accuracy_plot(self.training_thread.final_val_accuracies)

    # Call the training thread for the AlexNet model
    def train_alexnet(self, train_loader, test_loader, epochs):
        self.stacked_widget.setCurrentWidget(self.train_progress_page)
        self.training_thread = TrainingThreadAlexNet(train_loader, test_loader, epochs, train_test_ratio=self.train_test_ratio_slider.value() / 100.0)
        self.train_loader = train_loader
        self.training_thread.progress_updated.connect(self.train_progress_bar.setValue)
        self.training_thread.training_stopped.connect(self.on_training_stopped)
        self.training_thread.epoch_updated.connect(self.update_epoch_label)
        self.training_thread.batch_updated.connect(self.update_batch_label)
        self.training_thread.train_loss_updated.connect(self.update_train_loss_plot)
        self.training_thread.val_accuracy_updated.connect(self.update_val_accuracy_plot)
        self.training_thread.start()

    # Call the training thread for the InceptionV1 model
    def train_inceptionV1(self, train_loader, test_loader, epochs):
        self.stacked_widget.setCurrentWidget(self.train_progress_page)
        self.training_thread = TrainingThreadInceptionV1(train_loader, test_loader, epochs, train_test_ratio=self.train_test_ratio_slider.value() / 100.0)
        self.train_loader = train_loader
        self.training_thread.progress_updated.connect(self.train_progress_bar.setValue)
        self.training_thread.training_stopped.connect(self.on_training_stopped)
        self.training_thread.epoch_updated.connect(self.update_epoch_label)
        self.training_thread.batch_updated.connect(self.update_batch_label)
        self.training_thread.train_loss_updated.connect(self.update_train_loss_plot)
        self.training_thread.val_accuracy_updated.connect(self.update_val_accuracy_plot)
        self.training_thread.start()

    # Call the training thread for the ResNet model
    def train_resnet(self, train_loader, test_loader, epochs):
        self.stacked_widget.setCurrentWidget(self.train_progress_page)
        self.training_thread = TrainingThreadResnet(train_loader, test_loader, epochs, train_test_ratio=self.train_test_ratio_slider.value() / 100.0)
        self.train_loader = train_loader
        self.training_thread.progress_updated.connect(self.train_progress_bar.setValue)
        self.training_thread.training_stopped.connect(self.on_training_stopped)
        self.training_thread.epoch_updated.connect(self.update_epoch_label)
        self.training_thread.batch_updated.connect(self.update_batch_label)
        self.training_thread.train_loss_updated.connect(self.update_train_loss_plot)
        self.training_thread.val_accuracy_updated.connect(self.update_val_accuracy_plot)
        self.training_thread.start()

    # Call the training thread for the DenseNet model
    def train_densenet(self, train_loader, test_loader, epochs):
        self.stacked_widget.setCurrentWidget(self.train_progress_page)
        self.training_thread = TrainingThreadDenseNet(train_loader, test_loader, epochs, train_test_ratio=self.train_test_ratio_slider.value() / 100.0)
        self.train_loader = train_loader
        self.training_thread.progress_updated.connect(self.train_progress_bar.setValue)
        self.training_thread.training_stopped.connect(self.on_training_stopped)
        self.training_thread.epoch_updated.connect(self.update_epoch_label)
        self.training_thread.batch_updated.connect(self.update_batch_label)
        self.training_thread.train_loss_updated.connect(self.update_train_loss_plot)
        self.training_thread.val_accuracy_updated.connect(self.update_val_accuracy_plot)
        self.training_thread.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HandSeals()
    window.show()
    sys.exit(app.exec_())
