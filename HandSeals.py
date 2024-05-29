import sys
import csv
import numpy as np
import cv2
from dataloader import DataLoader
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar, QFileDialog, QScrollArea, QLabel, QGridLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class HandSeals(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Hand Seals')
        self.resize(1000, 800)
        self.images = []  # This will hold the loaded images
        self.current_image_index = 0  # Index to track loading progress
        self.setupUI()

    def setupUI(self):
        self.vertical_layout = QVBoxLayout(self)
        self.horizontal_layout = QHBoxLayout()

        # Buttons
        self.init_buttons()

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.horizontal_layout.addWidget(self.progress_bar)

        # Image Scroll Area
        self.image_scroll_area = QScrollArea(self)
        self.image_scroll_area.setWidgetResizable(True)
        self.image_container = QWidget()
        self.image_layout = QGridLayout(self.image_container)
        self.image_scroll_area.setWidget(self.image_container)
        self.vertical_layout.addLayout(self.horizontal_layout)
        self.vertical_layout.addWidget(self.image_scroll_area)

        # Connect the scroll event
        self.image_scroll_area.verticalScrollBar().valueChanged.connect(self.check_scroll)

    def init_buttons(self):
        self.load_data_button = QPushButton('Load Data')
        self.load_data_button.clicked.connect(self.button1_clicked)
        self.horizontal_layout.addWidget(self.load_data_button)

        self.view_button = QPushButton('View')
        self.view_button.clicked.connect(self.button2_clicked)
        self.horizontal_layout.addWidget(self.view_button)

        self.train_button = QPushButton('Train')
        self.train_button.clicked.connect(self.button3_clicked)
        self.horizontal_layout.addWidget(self.train_button)

        self.test_button = QPushButton('Test')
        self.test_button.clicked.connect(self.button4_clicked)
        self.horizontal_layout.addWidget(self.test_button)

    def button1_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.load_data(file_path)

    def load_data(self, file_path):
        self.data_loader = DataLoader(file_path)
        self.data_loader.dataLoaded.connect(self.on_data_loaded)
        self.data_loader.loadError.connect(self.on_data_load_error)
        self.data_loader.progressUpdated.connect(self.progress_bar.setValue)
        self.data_loader.start()

    def on_data_loaded(self, images):
        self.images = images
        self.progress_bar.setValue(100)

    def on_data_load_error(self, error):
        print(f"Error loading data: {error}")
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)  
        self.vertical_layout.addWidget(self.progress_bar)

        self.spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.vertical_layout.addItem(self.spacer)
        
        self.setLayout(self.vertical_layout)
        self.show()

    def button2_clicked(self):
        if self.images:
            self.current_image_index = 0
            self.clear_layout(self.image_layout)
            self.add_images_incrementally()

    def add_images_incrementally(self):
        batch_size = 100  # Load 100 images per batch
        start_index = self.current_image_index
        end_index = min(start_index + batch_size, len(self.images))
        for i in range(start_index, end_index):
            label, pixels = self.images[i]
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
            if self.current_image_index < len(self.images):
                self.add_images_incrementally()

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def button3_clicked(self):
        print("Training...")

    def button4_clicked(self):
        print("Testing...")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HandSeals()
    window.show()
    sys.exit(app.exec_())
