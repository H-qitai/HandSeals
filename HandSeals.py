import sys
import csv
import numpy as np
import cv2
import time
from dataloader import DataLoader
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar, QFileDialog, QScrollArea, QLabel, QGridLayout, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

class DataLoader(QThread):
    dataLoaded = pyqtSignal(list)
    loadError = pyqtSignal(str)
    progressUpdated = pyqtSignal(int)
    timeLeftUpdated = pyqtSignal(int, int)  # minutes, seconds

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self._is_running = True

    def run(self):
        try:
            images = []
            with open(self.file_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header row
                total_rows = sum(1 for row in reader)
                csvfile.seek(0)
                next(reader)  # Skip header row again
                start_time = time.time()

                for i, row in enumerate(reader):
                    if not self._is_running:
                        break
                    label = row[0]
                    pixels = np.array(row[1:], dtype=np.uint8).reshape((28, 28))
                    images.append((label, pixels))
                    elapsed_time = time.time() - start_time
                    rows_left = total_rows - (i + 1)
                    time_per_row = elapsed_time / (i + 1)
                    estimated_time_left = time_per_row * rows_left
                    minutes_left = int(estimated_time_left // 60)
                    seconds_left = int(estimated_time_left % 60)
                    self.progressUpdated.emit(int((i + 1) / total_rows * 100))
                    self.timeLeftUpdated.emit(minutes_left, seconds_left)
            if self._is_running:
                self.dataLoaded.emit(images)
        except Exception as e:
            self.loadError.emit(str(e))

    def stop(self):
        self._is_running = False

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

        # Progress Bar and Time Left
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.horizontal_layout.addWidget(self.progress_bar)
        self.time_left_label = QLabel("Time left: 00:00")
        self.horizontal_layout.addWidget(self.time_left_label)

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

    def on_data_loaded(self, images):
        self.images = images
        self.progress_bar.setValue(100)

    def on_data_load_error(self, error):
        print(f"Error loading data: {error}")

    def update_time_left(self, minutes, seconds):
        self.time_left_label.setText(f"Time left: {minutes:02}:{seconds:02}")

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
