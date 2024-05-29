import sys
import csv
import numpy as np
import cv2
import time
from dataloader import DataLoader  # Assuming this is implemented elsewhere
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar, QFileDialog, QScrollArea, QLabel, QGridLayout, QLineEdit, QComboBox, QSlider, QStackedWidget
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class HandSeals(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Hand Seals')
        self.resize(1000, 800)
        self.images = []  # This will hold the loaded images
        self.filtered_images = []  # This will hold the filtered images
        self.current_image_index = 0  # Index to track loading progress
        self.data_loader = None  # Thread to load data
        self.search_bar_added = False  # Flag to check if search bar is added
        self.initUI()

    def initUI(self):
        self.vertical_layout = QVBoxLayout(self)
        self.horizontal_layout = QHBoxLayout()

        # Buttons
        self.init_buttons()

        # Search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search by label...")
        self.search_bar.textChanged.connect(self.filter_images)

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

        # Image Scroll Area
        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidgetResizable(True)
        self.image_container = QWidget()
        self.image_layout = QGridLayout(self.image_container)
        self.image_scroll_area.setWidget(self.image_container)

        self.view_layout.addWidget(self.search_bar)
        self.view_layout.addWidget(self.image_scroll_area)

        self.stacked_widget.addWidget(self.view_page)

        # Create train page
        self.train_page = QWidget()
        self.train_layout = QVBoxLayout(self.train_page)
        self.train_settings()

        self.stacked_widget.addWidget(self.train_page)

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
        self.horizontal_layout.addWidget(self.stop_button)

    def button1_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.load_data(file_path)

    def load_data(self, file_path):
        self.data_loader = DataLoader(file_path)
        self.data_loader.dataLoaded.connect(self.on_data_loaded)
        self.data_loader.loadError.connect(self.on_data_load_error)
        self.data_loader.progressUpdated.connect(self.progress_bar.setValue)
        self.data_loader.timeLeftUpdated.connect(self.update_time_left)
        self.data_loader.start()

    def stop_loading(self):
        if self.data_loader and self.data_loader.isRunning():
            self.data_loader.stop()
            self.progress_bar.setValue(0)
            self.time_left_label.setText("Time left: 00:00")
        else:
            print("No data loading in progress")

    def on_data_loaded(self, images):
        self.images = images
        self.filtered_images = images  # Initially, all images are shown
        self.progress_bar.setValue(100)

    def on_data_load_error(self, error):
        print(f"Error loading data: {error}")

    def update_time_left(self, minutes, seconds):
        self.time_left_label.setText(f"Time left: {minutes:02}:{seconds:02}")

    def show_view_page(self):
        self.stacked_widget.setCurrentWidget(self.view_page)
        if self.images:
            self.current_image_index = 0
            self.clear_layout(self.image_layout)
            self.add_images_incrementally()

    def add_images_incrementally(self):
        batch_size = 100  # Load 100 images per batch
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
        image = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        bytes_per_line = 3 * array.shape[1]
        q_image = QImage(image.data, array.shape[1], array.shape[0], bytes_per_line, QImage.Format_RGB888)
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
            self.filtered_images = [img for img in self.images if search_text in img[0].lower()]
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
        self.train_layout.addWidget(self.start_training_button)

        self.train_layout.addSpacing(30)

        self.model_label = QLabel("Select Model")
        self.model_label.setAlignment(Qt.AlignCenter)
        self.model_label.setFont(font)
        self.train_layout.addWidget(self.model_label)

        self.train_layout.addSpacing(20)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["AlexNet", "ResNet", "Inception v3"])
        self.train_layout.addWidget(self.model_combo)

        self.train_layout.addSpacing(60)

        self.train_test_ratio_label = QLabel("Train/Test Ratio")
        self.train_test_ratio_label.setAlignment(Qt.AlignCenter)
        self.train_test_ratio_label.setFont(font)
        self.train_layout.addWidget(self.train_test_ratio_label)

        self.train_layout.addSpacing(20)

        self.train_test_ratio_slider = QSlider(Qt.Horizontal)
        self.train_test_ratio_slider.setMinimum(1)  # Set the minimum value
        self.train_test_ratio_slider.setMaximum(100)  # Set the maximum value
        self.train_test_ratio_slider.setValue(50)  # Set the default value
        self.train_test_ratio_slider.setMinimumHeight(30)
        self.train_test_ratio_slider.setMinimumWidth(200)
        self.train_test_ratio_slider.valueChanged.connect(self.update_train_test_ratio_label)  # Connect the signal
        self.train_layout.addWidget(self.train_test_ratio_slider)

        self.train_test_ratio_value_label = QLabel("50")  # Default value display
        self.train_test_ratio_value_label.setAlignment(Qt.AlignCenter)
        self.train_layout.addWidget(self.train_test_ratio_value_label)

        self.train_layout.addSpacing(60)

        self.batch_size_label = QLabel("Batch Size")
        self.batch_size_label.setAlignment(Qt.AlignCenter)
        self.batch_size_label.setFont(font)
        self.train_layout.addWidget(self.batch_size_label)

        self.train_layout.addSpacing(20)

        self.batch_size_slider = QSlider(Qt.Horizontal)
        self.batch_size_slider.setMinimum(1)  # Set the minimum value
        self.batch_size_slider.setMaximum(200)  # Set the maximum value
        self.batch_size_slider.setValue(100)  # Set the default value
        self.batch_size_slider.setMinimumHeight(30)
        self.batch_size_slider.setMinimumWidth(200)
        self.batch_size_slider.valueChanged.connect(self.update_batch_size_label)  # Connect the signal
        self.train_layout.addWidget(self.batch_size_slider)

        self.batch_size_value_label = QLabel("100")  # Default value display
        self.batch_size_value_label.setAlignment(Qt.AlignCenter)
        self.train_layout.addWidget(self.batch_size_value_label)

        self.train_layout.addSpacing(60)

        self.epochs_label = QLabel("Epochs")
        self.epochs_label.setAlignment(Qt.AlignCenter)
        self.epochs_label.setFont(font)
        self.train_layout.addWidget(self.epochs_label)

        self.train_layout.addSpacing(20)

        self.epochs_slider = QSlider(Qt.Horizontal)
        self.epochs_slider.setMinimum(1)  # Set the minimum value
        self.epochs_slider.setMaximum(100)  # Set the maximum value
        self.epochs_slider.setValue(30)  # Set the default value
        self.epochs_slider.setMinimumHeight(30)
        self.epochs_slider.setMinimumWidth(200)
        self.epochs_slider.valueChanged.connect(self.update_epochs_label)  # Connect the signal
        self.train_layout.addWidget(self.epochs_slider)

        self.epochs_value_label = QLabel("30")  # Default value display
        self.epochs_value_label.setAlignment(Qt.AlignCenter)
        self.train_layout.addWidget(self.epochs_value_label)

        self.train_layout.addSpacing(60)


    def start_training(self):
        # Fetch selected model from the combo box
        selected_model = self.model_combo.currentText()
        print(f'Training Started using {selected_model}')
        # Implement the training logic based on selected model
        if selected_model == 'AlexNet':
            self.train_alexnet()
        elif selected_model == 'ResNet':
            self.train_resnet()
        elif selected_model == 'Inception v3':
            self.train_inception()

    def train_alexnet(self):
        print("Training AlexNet...")

    def train_resnet(self):
        print("Training ResNet...")

    def train_inception(self):
        print("Training Inception v3...")

    def button4_clicked(self):
        print("Testing...")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HandSeals()
    window.show()
    sys.exit(app.exec_())
