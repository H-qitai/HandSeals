# dataloader.py
import csv
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

class DataLoader(QThread):
    dataLoaded = pyqtSignal(list)
    loadError = pyqtSignal(str)
    progressUpdated = pyqtSignal(int)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            images = []
            with open(self.file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header row
                total_rows = sum(1 for row in reader)
                csvfile.seek(0)
                next(reader)  # Skip header again to reset
                for index, row in enumerate(reader):
                    if row:
                        label = row[0]
                        pixels = np.array(row[1:], dtype=np.uint8).reshape((28, 28))
                        images.append((label, pixels))
                        self.progressUpdated.emit(int((index + 1) / total_rows * 100))
            self.dataLoaded.emit(images)
        except Exception as e:
            self.loadError.emit(str(e))
