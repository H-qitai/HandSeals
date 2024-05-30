# dataloader.py
import csv
import numpy as np
import time
from PyQt5.QtCore import QThread, pyqtSignal

class CSVDataLoader(QThread):
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

                # Remove or update the bounds check
                for i, row in enumerate(reader):
                    if not self._is_running:
                        break
                    label = int(row[0])
                    if label < 0 or label >= 36:  # Update the range if you have a different number of classes
                        raise ValueError(f"Label out of bounds: {label}")
                    pixels = np.array(row[1:], dtype=np.uint8).reshape((1, 28, 28))
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
