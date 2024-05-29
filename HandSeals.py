import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSpacerItem, QSizePolicy, QLabel, QComboBox, QSlider, QPushButton
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class HandSeals(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Hand Seals')
        self.move(50, 50)
        self.resize(1000, 800)

        self.vertical_layout = QVBoxLayout()
        self.horizontal_layout = QHBoxLayout()

        #Button 1
        b1 = QtWidgets.QPushButton(self)
        b1.setText('Load Data')
        b1.clicked.connect(self.button1_clicked)
        self.horizontal_layout.addWidget(b1)

        #Button 2
        b2 = QtWidgets.QPushButton(self)
        b2.setText('View')
        b2.clicked.connect(self.button2_clicked)
        self.horizontal_layout.addWidget(b2)

        #Button 3
        b3 = QtWidgets.QPushButton(self)
        b3.setText('Train')
        b3.clicked.connect(self.button3_clicked)
        self.horizontal_layout.addWidget(b3)

        #Button 4
        b4 = QtWidgets.QPushButton(self)
        b4.setText('Test')
        b4.clicked.connect(self.button4_clicked)
        self.horizontal_layout.addWidget(b4)

        self.vertical_layout.addLayout(self.horizontal_layout)
        self.spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.vertical_layout.addItem(self.spacer)
        
        self.setLayout(self.vertical_layout)
        self.show()

    def button1_clicked(self):
        print('Data Loaded')
    
    def button2_clicked(self):
        print('Viewing Data')
    
    def button3_clicked(self):

        for i in reversed(range(self.vertical_layout.count())):
            item = self.vertical_layout.itemAt(i)
            if item != self.horizontal_layout:
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                elif item.spacerItem():
                    self.vertical_layout.removeItem(item)

        self.training_settings()

    
    def training_settings(self):

        font = QFont("Arial", 10, QFont.Bold)
        self.vertical_layout.addSpacing(30)


        self.start_training_button = QPushButton("Start Training")
        self.start_training_button.clicked.connect(self.start_training)
        self.vertical_layout.addWidget(self.start_training_button)

        self.vertical_layout.addSpacing(30)
        
        self.model_label = QLabel("Select Model")
        self.model_label.setAlignment(Qt.AlignCenter)
        self.model_label.setFont(font)
        self.vertical_layout.addWidget(self.model_label)

        self.vertical_layout.addSpacing(20)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["AlexNet", "ResNet", "Inception v3"])
        self.vertical_layout.addWidget(self.model_combo)

        self.vertical_layout.addSpacing(60)

        self.train_test_ratio_label = QLabel("Train/Test Ratio")
        self.train_test_ratio_label.setAlignment(Qt.AlignCenter)
        self.train_test_ratio_label.setFont(font)
        self.vertical_layout.addWidget(self.train_test_ratio_label)
        
        self.vertical_layout.addSpacing(20)

        self.train_test_ratio_slider = QSlider(Qt.Horizontal)
        self.train_test_ratio_slider.setMinimumHeight(30)
        self.train_test_ratio_slider.setMinimumWidth(200)
        self.vertical_layout.addWidget(self.train_test_ratio_slider)

        self.vertical_layout.addSpacing(60)

        self.batch_size_label = QLabel("Batch Size")
        self.batch_size_label.setAlignment(Qt.AlignCenter)
        self.batch_size_label.setFont(font)
        self.vertical_layout.addWidget(self.batch_size_label)
        
        self.vertical_layout.addSpacing(20)

        self.batch_size_slider = QSlider(Qt.Horizontal)
        self.batch_size_slider.setMinimumHeight(30)
        self.batch_size_slider.setMinimumWidth(200)
        self.vertical_layout.addWidget(self.batch_size_slider)

        self.vertical_layout.addSpacing(60)

        self.epochs_label = QLabel("Epochs")
        self.epochs_label.setAlignment(Qt.AlignCenter)
        self.epochs_label.setFont(font)
        self.model_label.setFont(font)
        self.vertical_layout.addWidget(self.epochs_label)

        self.vertical_layout.addSpacing(20)

        self.epochs_slider = QSlider(Qt.Horizontal)
        self.epochs_slider.setMinimumHeight(30)
        self.epochs_slider.setMinimumWidth(200)
        self.vertical_layout.addWidget(self.epochs_slider)

        self.vertical_layout.addSpacing(60)

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
        print('Testing Model')

    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = HandSeals()
    sys.exit(app.exec_())