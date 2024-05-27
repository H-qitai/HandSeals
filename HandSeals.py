import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QSpacerItem, QSizePolicy
from PyQt5 import QtWidgets

class HandSeals(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Hand Seals')
        self.move(50, 50)
        self.resize(1500, 800)

        vertical_layout = QVBoxLayout()
        horizontal_layout = QHBoxLayout()

        b1 = QtWidgets.QPushButton(self)
        b1.setText('Load Data')
        b1.clicked.connect(self.button1_clicked)
        horizontal_layout.addWidget(b1)

        b2 = QtWidgets.QPushButton(self)
        b2.setText('View')
        b2.clicked.connect(self.button2_clicked)
        horizontal_layout.addWidget(b2)

        b3 = QtWidgets.QPushButton(self)
        b3.setText('Train')
        b3.clicked.connect(self.button3_clicked)
        horizontal_layout.addWidget(b3)

        b4 = QtWidgets.QPushButton(self)
        b4.setText('Test')
        b4.clicked.connect(self.button4_clicked)
        horizontal_layout.addWidget(b4)

        vertical_layout.addLayout(horizontal_layout)
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        vertical_layout.addItem(spacer)
        
        self.setLayout(vertical_layout)
        self.show()

    def button1_clicked(self):
        print('Data Loaded')
    
    def button2_clicked(self):
        print('Viewing Data')
    
    def button3_clicked(self):
        print('Training Model')

    def button4_clicked(self):
        print('Testing Model')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = HandSeals()
    sys.exit(app.exec_())